"""Verified adapter between rl/ and pyclashbot's actual API.

Audited against the repo on-disk:

    pyclashbot/emulators/memu.py        MemuEmulatorController(logger, render_mode, debug_mode)
    pyclashbot/emulators/bluestacks.py  BlueStacksEmulatorController(logger, ...)
    pyclashbot/emulators/base.py        BaseEmulatorController.{click, swipe, screenshot}
    pyclashbot/detection/image_rec.py   find_image(image, folder, tolerance, subcrop)
    pyclashbot/bot/nav.py               check_if_in_battle, wait_for_battle_start,
                                        check_if_battle_has_ended, check_for_post_battle_button,
                                        check_if_on_clash_main_menu, select_mode
    pyclashbot/bot/fight.py             start_fight, get_to_main_after_fight, HAND_CARDS_COORDS

The key surprise is that the emulator is not a set of free functions but a
stateful controller object that MUST be instantiated with a Logger. Detection
helpers in nav.py take that same controller and call .screenshot() themselves
(so we rarely need to pass a frame around — we pass the emulator).

Screen geometry is fixed at 419x633 by MEMU_CONFIGURATION in memu.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from pyclashbot.bot.fight import (
    ELIXIR_COLOR,
    ELIXIR_COORDS,
    HAND_CARDS_COORDS,
    get_to_main_after_fight,
    start_fight,
)
from pyclashbot.bot.nav import (
    check_if_battle_has_ended,
    check_if_battle_mode_is_selected,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    select_mode,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.utils.cancellation import interruptible_sleep
from pyclashbot.utils.logger import Logger
from pyclashbot.utils.platform import is_macos

if TYPE_CHECKING:
    from pyclashbot.emulators.base import BaseEmulatorController

FightMode = Literal["Classic 1v1", "Classic 2v2", "Trophy Road"]
EmulatorType = Literal["bluestacks", "memu"]

# Default battle mode. Trophy Road matches are typically shorter and
# always available on any account, so it's the cheapest to train on.
DEFAULT_MODE: FightMode = "Trophy Road"

SCREEN_W = 419
SCREEN_H = 633

# BlueStacks graphics_renderer codes (see bluestacks.py::_normalize_renderer).
# On macOS you almost always want "vlcn" (Vulkan); "gl" is a fallback; "dx"
# is Windows-only and will silently fall back to the platform default.
DEFAULT_BLUESTACKS_RENDER: dict[str, str] = {"graphics_renderer": "vlcn"}

_emulator: BaseEmulatorController | None = None
_logger: Logger | None = None


def _default_emulator_type() -> EmulatorType:
    """MEmu is Windows-only; default to BlueStacks on macOS."""
    return "bluestacks" if is_macos() else "memu"


def get_logger() -> Logger:
    """Lazily construct a shared Logger instance.

    The pyclashbot Logger has no-arg construction and backs a thread-safe
    in-memory stats dict; it is safe to share across the RL loop.
    """
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger


def get_emulator(
    emulator_type: EmulatorType | None = None,
    render_mode: str | None = None,
    render_settings: dict[str, Any] | None = None,
    debug_mode: bool = False,
) -> BaseEmulatorController:
    """Lazily construct and cache the emulator controller.

    Args:
        emulator_type: "bluestacks" (macOS + Windows) or "memu" (Windows only).
            Defaults to "bluestacks" on macOS, "memu" on Windows.
        render_mode: MEmu-only. "directx" or "opengl". Ignored for BlueStacks.
        render_settings: BlueStacks-only. Dict with key "graphics_renderer";
            values "vlcn" (Vulkan, macOS default), "gl" (OpenGL), or "dx"
            (DirectX, Windows-only). Ignored for MEmu. If omitted,
            DEFAULT_BLUESTACKS_RENDER is used.
        debug_mode: MEmu-only flag that skips configure/restart so you can
            attach to an already-running VM for iterative development.
    """
    global _emulator
    if _emulator is not None:
        return _emulator

    chosen = emulator_type or _default_emulator_type()
    logger = get_logger()

    if chosen == "memu":
        from pyclashbot.emulators.memu import MemuEmulatorController

        _emulator = MemuEmulatorController(
            logger,
            render_mode=render_mode or "directx",
            debug_mode=debug_mode,
        )
    elif chosen == "bluestacks":
        from pyclashbot.emulators.bluestacks import BlueStacksEmulatorController

        settings = render_settings or DEFAULT_BLUESTACKS_RENDER
        if "graphics_renderer" not in settings:
            # BlueStacks indexes this key unconditionally — fail loudly
            # rather than letting the controller raise deep in its boot.
            raise ValueError(
                "render_settings must contain a 'graphics_renderer' key "
                "(e.g. {'graphics_renderer': 'vlcn'})"
            )
        _emulator = BlueStacksEmulatorController(logger, render_settings=settings)
    else:
        raise ValueError(f"Unknown emulator type: {chosen}")
    return _emulator


def set_emulator(emulator: BaseEmulatorController) -> None:
    """Inject an externally-created controller (useful for tests)."""
    global _emulator
    _emulator = emulator


def get_screen() -> np.ndarray:
    """Return the current emulator frame as a (633, 419, 3) BGR array."""
    return get_emulator().screenshot()


def click(x: int, y: int, clicks: int = 1) -> None:
    """Tap the emulator screen at pixel (x, y)."""
    get_emulator().click(x, y, clicks=clicks, interval=0.1)


def swipe(x1: int, y1: int, x2: int, y2: int) -> None:
    get_emulator().swipe(x1, y1, x2, y2)


def is_in_battle() -> bool:
    """Pixel-based scoreboard detection from pyclashbot.bot.nav."""
    return bool(check_if_in_battle(get_emulator()))


def is_on_main_menu() -> bool:
    return bool(check_if_on_clash_main_menu(get_emulator()))


def is_battle_over() -> bool:
    """True on either the main menu, trophy reward screen, or post-battle OK."""
    return bool(check_if_battle_has_ended(get_emulator()))


def start_battle(mode: FightMode = DEFAULT_MODE, start_timeout: int = 120) -> bool:
    """Navigate from the main menu to an active battle.

    Mirrors what fight.do_fight_state does on entry, plus a verify-then-
    retry gate around mode selection: we never click the Battle button
    until check_if_battle_mode_is_selected confirms the requested mode
    is the highlighted one. This kills the failure mode where
    select_mode silently leaves a different mode selected (e.g. Merge
    Tactics) and the bot starts the wrong match.
    """
    emulator = get_emulator()
    logger = get_logger()

    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        return False

    # Verify-and-retry: click Battle ONLY once the requested mode is the
    # confirmed-selected mode. Three attempts before giving up — the
    # caller (env.reset) escalates further on False.
    mode_confirmed = False
    for _ in range(3):
        if check_if_battle_mode_is_selected(emulator, mode):
            mode_confirmed = True
            break
        if not select_mode(emulator, mode):
            # select_mode failed (couldn't find the icon); send back-key
            # in case we're trapped in the mode-selection scroll panel,
            # then retry from the top.
            emulator.send_back_key()
            interruptible_sleep(1)
    if not mode_confirmed:
        return False

    if not start_fight(emulator, logger, mode):
        return False
    return wait_for_battle_start(emulator, logger, timeout=start_timeout)


def return_to_main_menu() -> bool:
    """After an episode ends, dismiss post-battle screens back to the main menu."""
    emulator = get_emulator()
    logger = get_logger()
    return bool(get_to_main_after_fight(emulator, logger))


__all__ = [
    "ARENA_MIDLINE_Y",
    "HAND_CARDS_COORDS",
    "NUM_CARD_IDS",
    "SCREEN_H",
    "SCREEN_W",
    "TROOPS_HEATMAP_CHANNELS",
    "TROOPS_HEATMAP_H",
    "TROOPS_HEATMAP_SHAPE",
    "TROOPS_HEATMAP_W",
    "Side",
    "TroopDetection",
    "classify_detection",
    "click",
    "detect_troops",
    "get_card_id_map",
    "get_emulator",
    "get_logger",
    "get_roboflow_client",
    "get_screen",
    "is_battle_over",
    "is_in_battle",
    "is_on_main_menu",
    "rasterize_detections",
    "read_elixir",
    "read_hand",
    "return_to_main_menu",
    "set_emulator",
    "start_battle",
    "summarize_detections",
    "swipe",
]


# ----------------------------------------------------------------------
# In-battle perception helpers (cards + elixir)
# ----------------------------------------------------------------------
# Cards are recognised via pyclashbot's hand-coded color-histogram
# classifier in pyclashbot.bot.card_detection. It covers 77 known cards
# plus "UNKNOWN"; we expose a stable alphabetical {name -> int id}
# mapping so the RL observation can use the id directly.

_CARD_ID_MAP: dict[str, int] | None = None


def get_card_id_map() -> dict[str, int]:
    """Return the cached {card_name -> int_id} mapping.

    id 0 is reserved for UNKNOWN; the cards pyclashbot knows about get
    ids 1..N in sorted alphabetical order (stable across runs).
    """
    global _CARD_ID_MAP
    if _CARD_ID_MAP is None:
        from pyclashbot.bot import card_detection as _cd

        names = sorted(_cd.card_color_data.keys())
        mapping: dict[str, int] = {"UNKNOWN": 0}
        for i, name in enumerate(names, start=1):
            mapping[name] = i
        _CARD_ID_MAP = mapping
    return _CARD_ID_MAP


# Total vocabulary size (known cards + UNKNOWN). Computed once at import
# time from pyclashbot's classifier; keeps the RL observation space in
# sync with pyclashbot even if more cards are added upstream.
NUM_CARD_IDS: int = len(get_card_id_map())


def read_hand(frame: np.ndarray) -> tuple[list[int], list[bool]]:
    """Identify the 4 hand cards and their affordability from a frame.

    Returns:
        (card_ids, affordable_flags) each a list of 4 entries.
        card_ids are integers from get_card_id_map().
        affordable_flags are booleans: True iff the slot shows the
        purple elixir-cost icon (i.e. enough elixir to play it).

    Implementation note: pyclashbot's `find_closest_card` /
    `get_all_pixel_data` read from a module-level `battle_iar` variable
    rather than from their `emulator` argument (which is ignored in
    practice). We set that variable to the passed-in frame to avoid an
    extra screenshot round-trip per step.
    """
    from pyclashbot.bot import card_detection as _cd

    id_map = get_card_id_map()

    _cd.battle_iar = frame

    card_ids: list[int] = []
    affordable: list[bool] = []
    for i in range(4):
        try:
            name = _cd.find_closest_card(_cd.get_all_pixel_data(None, i))
        except Exception:
            name = "UNKNOWN"
        card_ids.append(id_map.get(name or "UNKNOWN", 0))

        try:
            x_coords, y_coords = _cd.card_coords[i]
            iar_pixels = frame[np.ix_(y_coords, x_coords)]
            purple = np.all(np.abs(iar_pixels - _cd.purple_color) <= 30, axis=-1)
            affordable.append(bool(np.sum(purple) >= 26))
        except Exception:
            affordable.append(False)

    return card_ids, affordable


def read_elixir(frame: np.ndarray) -> int:
    """Return current elixir [0, 10] by scanning ELIXIR_COORDS.

    Mirrors pyclashbot.bot.fight.count_elixer: the bar fills
    left-to-right, so we count contiguously-matching slots from slot 1.
    Returns the previous count logic: highest filled slot.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return 0

    h, w = frame.shape[:2]
    target = np.array(ELIXIR_COLOR, dtype=np.int16)
    count = 0
    for y, x in ELIXIR_COORDS:
        if y >= h or x >= w:
            break
        px = frame[y, x].astype(np.int16)
        if np.all(np.abs(px - target) <= 50):
            count += 1
        else:
            break
    return count


# ----------------------------------------------------------------------
# Roboflow-hosted troop detection (optional)
# ----------------------------------------------------------------------
# We call a public Roboflow Universe model directly via client.infer(),
# bypassing the workflow abstraction entirely. Universe models are
# globally accessible to any account with a valid API key, so there is
# no fork / workspace gymnastics needed.
#
# Default model: nejc-zavodnik/clash-royale-troop-detection (version 1)
#   https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection
#
# To avoid hitting Roboflow's cloud endpoint on every step, point at a
# local inference server. Two ways to run one:
#   * GUI: download the Roboflow Inference desktop app from
#       https://github.com/roboflow/inference/releases/latest
#     and launch it (menu-bar icon, listens on http://localhost:9001).
#   * CLI: `uv sync --group roboflow` then `inference server start`.
#
# Environment variables consumed here:
#     ROBOFLOW_API_KEY        — required (key from app.roboflow.com/settings/api)
#     ROBOFLOW_TROOP_MODEL_ID — optional, defaults to the Universe model above
#     ROBOFLOW_API_URL        — optional, defaults to http://localhost:9001


DEFAULT_ROBOFLOW_API_URL = "http://localhost:9001"

# The format expected by client.infer(model_id=...) is strictly
# "<project-slug>/<version>" — two parts, NOT three. The SDK source
# rejects anything else with "InvalidModelIdentifier". Universe project
# slugs have random suffixes (e.g. `-of3d3`, `-vop4y`) so two parts are
# enough to uniquely identify any public Universe model globally.
#
# Known-working public Clash Royale troop detection models on Universe.
# Pick whichever tests best for your account + BlueStacks resolution
# (override at runtime with the ROBOFLOW_TROOP_MODEL_ID env var).
#
#   clash-royale-xy2jw/2  ← current default
#     https://universe.roboflow.com/workspace-mck69/clash-royale-xy2jw/model/2
#
#   clash-royale-of3d3/1
#     https://universe.roboflow.com/clashroyale/clash-royale-of3d3
#     972 images, 72 classes, large but old (2022).
#
#   clash-royale-ut3g8/1
#     https://universe.roboflow.com/yolotrain-c7s7v/clash-royale-ut3g8
#
# To use a different model, get its ID from the Universe page:
#   1. Open the model page in a browser.
#   2. Click "Deploy" (top right) -> Python tab.
#   3. The "Copy Model ID" button gives you exactly "<slug>/<version>".
DEFAULT_TROOP_MODEL_ID = "clash-royale-xy2jw/2"

_ROBOFLOW_CLIENT: Any = None


@dataclass(frozen=True)
class TroopDetection:
    """A single bounding-box detection from a Roboflow workflow.

    Coordinates are in the same pixel space as the frame fed to
    `detect_troops` — i.e. the raw 419x633 BlueStacks screenshot.
    (x, y) is the *center* of the box, matching Roboflow's convention.
    """

    cls: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        """Convert center+size to (x0, y0, x1, y1) ints for drawing."""
        x0 = int(self.x - self.width / 2)
        y0 = int(self.y - self.height / 2)
        x1 = int(self.x + self.width / 2)
        y1 = int(self.y + self.height / 2)
        return x0, y0, x1, y1


def get_roboflow_client() -> Any:
    """Lazily construct and cache the InferenceHTTPClient.

    Raises RuntimeError with actionable messages if the SDK is not
    installed or the API key is missing; never fails silently.
    """
    global _ROBOFLOW_CLIENT
    if _ROBOFLOW_CLIENT is not None:
        return _ROBOFLOW_CLIENT

    try:
        from inference_sdk import InferenceHTTPClient  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        msg = (
            "inference-sdk is not installed. Run:\n"
            "    uv sync --group roboflow\n"
            "then start the local inference server:\n"
            "    inference server start"
        )
        raise RuntimeError(msg) from e

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY env var is not set. Get a key from "
            "https://app.roboflow.com/settings/api and export it."
        )

    api_url = os.environ.get("ROBOFLOW_API_URL", DEFAULT_ROBOFLOW_API_URL)
    _ROBOFLOW_CLIENT = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return _ROBOFLOW_CLIENT


def _extract_predictions(result: Any) -> list[dict[str, Any]]:
    """Pull the list of bbox predictions out of a client.infer() response.

    The infer() endpoint returns a single dict shaped like:

        {"predictions": [
             {"x": ..., "y": ..., "width": ..., "height": ...,
              "class": ..., "confidence": ...,  "class_id": ...},
             ...
         ],
         "image": {"width": ..., "height": ...},
         "time": ..., ...}

    Returns [] on any unrecognised shape so callers can degrade
    gracefully. Run `rl/tools/test_roboflow.py` to see the raw shape
    for your specific model.
    """
    if not isinstance(result, dict):
        return []
    preds = result.get("predictions")
    if not isinstance(preds, list):
        return []
    return [p for p in preds if isinstance(p, dict) and "class" in p and "confidence" in p]


def detect_troops(
    frame: np.ndarray,
    model_id: str | None = None,
    min_confidence: float = 0.25,
) -> list[TroopDetection]:
    """Run a Roboflow object-detection model on a BGR frame.

    Args:
        frame: (H, W, 3) BGR image from `get_screen()`. The numpy array
            is sent directly; the inference server handles encoding.
        model_id: overrides ROBOFLOW_TROOP_MODEL_ID env var. Format is
            "<workspace>/<project>/<version>", e.g.
            "nejc-zavodnik/clash-royale-troop-detection/1".
        min_confidence: drop detections weaker than this.

    Returns:
        List of TroopDetection in frame pixel coordinates.

    Raises:
        RuntimeError: if the SDK/server/key is unavailable, or if the
            model returns 404 (typically a typo in the model_id).
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    chosen_model = model_id or os.environ.get("ROBOFLOW_TROOP_MODEL_ID", DEFAULT_TROOP_MODEL_ID)

    if chosen_model.count("/") != 1:
        raise RuntimeError(
            f"Invalid model_id '{chosen_model}'. Expected exactly two parts "
            "('<project-slug>/<version>'), not three. "
            "Even for Universe models the SDK strips the workspace prefix — use the "
            "'Copy Model ID' button on the model's Universe page to get the right string."
        )

    client = get_roboflow_client()
    try:
        result = client.infer(frame, model_id=chosen_model)
    except Exception as e:
        status = getattr(e, "status_code", None)
        desc = str(getattr(e, "description", e))
        if status in (400, 404) or "Invalid Model ID" in desc:
            raise RuntimeError(
                f"Roboflow rejected model_id='{chosen_model}' (status={status}).\n"
                f"  api_message: {getattr(e, 'api_message', desc)}\n\n"
                "Causes (ranked by likelihood):\n"
                "  1. Project slug mistyped or the Universe page was deleted.\n"
                "  2. Version number doesn't exist (try /1 if you had /2).\n"
                "  3. Private model — your API key has no access.\n\n"
                "Fix: open the model's Universe page, click 'Deploy', and use the exact\n"
                "string under 'Copy Model ID' (should contain exactly one '/')."
            ) from e
        raise

    predictions = _extract_predictions(result)
    detections: list[TroopDetection] = []
    for p in predictions:
        try:
            conf = float(p.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < min_confidence:
            continue
        try:
            detections.append(
                TroopDetection(
                    cls=str(p["class"]),
                    confidence=conf,
                    x=float(p["x"]),
                    y=float(p["y"]),
                    width=float(p["width"]),
                    height=float(p["height"]),
                    raw=p,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return detections


# ----------------------------------------------------------------------
# Detection -> RL signal helpers
# ----------------------------------------------------------------------
# The arena is symmetric: the bottom half (y >= ARENA_MIDLINE_Y) is the
# agent's side, the top half is the opponent's. Most Roboflow troop-
# detection models include team info in the class name ("ally", "enemy",
# "blue", "red", "friendly", ...) but the convention varies model-to-
# model. We use a hybrid classifier: trust class-name keywords when they
# exist, fall back to y-position. Position is reliable because the camera
# never rotates and pyclashbot only ever fights on the bottom half.

ARENA_MIDLINE_Y = 300

# Heatmap geometry. NatureCNN (SB3's default image extractor for boxes
# with 3-d shape) requires inputs >= 36 px on each spatial dim, so 64x64
# is the smallest sensible size. Two channels: friendly / enemy.
TROOPS_HEATMAP_H = 64
TROOPS_HEATMAP_W = 64
TROOPS_HEATMAP_CHANNELS = 2
TROOPS_HEATMAP_SHAPE = (TROOPS_HEATMAP_H, TROOPS_HEATMAP_W, TROOPS_HEATMAP_CHANNELS)

Side = Literal["friendly", "enemy", "tower", "unknown"]

# Class-name keyword maps. Lowercased, substring-matched. Order matters:
# tower keywords are checked before team keywords so e.g.
# "Enemy King Tower" classifies as TOWER (not ENEMY troop). Tower
# detections are excluded from the heatmap on purpose — the policy
# already learns tower locations from the pixel input, and tower bboxes
# are huge so they would dominate the heatmap if included.
_TOWER_KEYWORDS = ("tower", "king", "princess")
_FRIENDLY_KEYWORDS = ("ally", "friend", "blue", "self", " my ", "my-", "my_")
_ENEMY_KEYWORDS = ("enemy", "opp", "red", "foe", "hostile")


def classify_detection(det: TroopDetection) -> Side:
    """Classify a detection as friendly troop / enemy troop / tower.

    Resolution order:
        1. Tower keywords in the class name -> TOWER (always, regardless
           of position). Tower bboxes are too large to be useful as
           "unit mass" signals.
        2. Team keywords in the class name (ally/enemy/blue/red/...) ->
           FRIENDLY or ENEMY. Trusts the model's labels.
        3. Fall back to y-coordinate vs ARENA_MIDLINE_Y: top half is
           enemy, bottom half is friendly. Reliable because the camera
           never rotates.
    """
    name = det.cls.lower()
    for kw in _TOWER_KEYWORDS:
        if kw in name:
            return "tower"
    for kw in _FRIENDLY_KEYWORDS:
        if kw in name:
            return "friendly"
    for kw in _ENEMY_KEYWORDS:
        if kw in name:
            return "enemy"
    return "enemy" if det.y < ARENA_MIDLINE_Y else "friendly"


def summarize_detections(
    detections: list[TroopDetection] | None,
) -> tuple[int, int]:
    """Return (friendly_troop_count, enemy_troop_count) for shaping.

    Towers are excluded from both counts. Returns (0, 0) for None /
    empty input so callers can use the result unconditionally.
    """
    if not detections:
        return 0, 0
    friendly = 0
    enemy = 0
    for det in detections:
        side = classify_detection(det)
        if side == "friendly":
            friendly += 1
        elif side == "enemy":
            enemy += 1
    return friendly, enemy


def rasterize_detections(
    detections: list[TroopDetection] | None,
    out_shape: tuple[int, int, int] = TROOPS_HEATMAP_SHAPE,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Render detections as a 2-channel uint8 heatmap.

    Channel 0 is friendly mass, channel 1 is enemy mass. Each non-tower
    detection is splatted as a filled circle at its scaled center, with
    radius proportional to the bbox size and intensity proportional to
    confidence. The whole map is then Gaussian-blurred so the policy
    can interpolate between adjacent positions instead of treating each
    detection as a delta.

    The output is HWC uint8 in [0, 255] — the same dtype/layout SB3
    expects for image observations, so it routes through NatureCNN
    automatically.
    """
    h, w, c = out_shape
    if not detections:
        return np.zeros((h, w, c), dtype=np.uint8)

    # Draw into per-channel contiguous buffers — cv2.circle rejects
    # non-contiguous numpy slices (e.g. heatmap[:, :, ch]).
    channels = [np.zeros((h, w), dtype=np.uint8) for _ in range(c)]

    sx = w / SCREEN_W
    sy = h / SCREEN_H

    for det in detections:
        side = classify_detection(det)
        if side == "friendly":
            ch = 0
        elif side == "enemy":
            ch = 1
        else:
            continue
        cx = int(round(det.x * sx))
        cy = int(round(det.y * sy))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        # Radius scales with bbox size in heatmap coords. Floor at 2 so
        # tiny detections still register; cap at 1/4 of frame so a
        # single huge bbox can't blanket the map.
        radius = int(max(2, min(min(h, w) // 4, ((det.width + det.height) / 4) * sx)))
        intensity = int(max(0, min(255, round(255 * det.confidence))))
        cv2.circle(channels[ch], (cx, cy), radius, (float(intensity),), thickness=-1)

    if blur_kernel > 1:
        for i in range(c):
            channels[i] = cv2.GaussianBlur(channels[i], (blur_kernel, blur_kernel), 0)
    return np.stack(channels, axis=-1)
