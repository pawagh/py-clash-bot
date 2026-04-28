"""Gymnasium environment wrapping a live Clash Royale emulator.

Runs in lockstep with pyclashbot: each step taps up to two pixels on the
emulator, waits FRAME_SKIP_SECONDS for the game to evolve, and then
returns a structured observation (downsampled grayscale frame + hand /
elixir state + optional Roboflow troop heatmap).

Observation layout (Dict):
    pixels           uint8  (OBS_SIZE, OBS_SIZE, 1)              arena screenshot
    troops           uint8  (TROOPS_HEATMAP_H, _W, 2)            friendly/enemy mass heatmap
    hand             int32  (4,)                                 pyclashbot card ids
    hand_affordable  uint8  (4,)                                 0/1 per slot
    elixir           uint8  (1,)                                 current 0..10

The `troops` channel is always present in the observation space so the
policy architecture is fixed; it is zero-filled when Roboflow detection
is disabled. When enabled (RL_USE_ROBOFLOW=1 or use_roboflow=True), one
detect_troops() call per step feeds *both* the heatmap and the reward
calculator — never two calls per step.

Every field is a gym.Box so VecFrameStack can stack all of them across
n_stack frames. Stacking the structured fields is intentional: it gives
the policy an implicit cycle history (what cards rotated out), the
elixir-regen trajectory, and short-term troop motion for free.

Timing notes:
    * The game runs in real time. FRAME_SKIP_SECONDS controls how much
      wall-clock the environment gives the game between decisions.
    * Roboflow inference adds ~250ms on a warm local server. Combined
      with the existing FRAME_SKIP, expect a step to take ~0.85s when
      detections are enabled.
    * pyclashbot bans `time.sleep` in favour of `interruptible_sleep`
      (see pyproject.toml's TID251 config) so the agent thread can be
      cancelled cleanly from the GUI.
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pyclashbot.utils.cancellation import interruptible_sleep
from rl.action_map import CARD_SLOTS, TOTAL_ACTIONS, decode
from rl.bridge import (
    DEFAULT_MODE,
    NUM_CARD_IDS,
    NUM_DETECTION_SLOTS,
    NUM_SIDES,
    SCREEN_H,
    SCREEN_W,
    TROOPS_HEATMAP_SHAPE,
    FightMode,
    TroopDetection,
    click,
    detect_troops,
    encode_detections,
    get_emulator,
    get_screen,
    is_in_battle,
    rasterize_detections,
    read_elixir,
    read_hand,
    return_to_main_menu,
    start_battle,
)
from rl.reward import RewardCalculator, StepInfo

OBS_SIZE = 128
PIXELS_SHAPE = (OBS_SIZE, OBS_SIZE, 1)

FRAME_SKIP_SECONDS = 0.6
CARD_SELECT_DELAY = 0.08
RESET_WAIT_TIMEOUT = 120


def _env_flag_default() -> bool:
    """Read RL_USE_ROBOFLOW; truthy values enable detection by default."""
    raw = os.environ.get("RL_USE_ROBOFLOW", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        mode: FightMode = DEFAULT_MODE,
        use_roboflow: bool | None = None,
    ) -> None:
        """Construct the env.

        Args:
            mode: which Clash Royale game mode to fight.
            use_roboflow: enable Roboflow troop detection per step. If
                None (default), reads the RL_USE_ROBOFLOW env var.
                Detection failures (server down, model error) are caught
                per-step and degrade gracefully to zeros — they will
                not crash training.
        """
        super().__init__()
        self.mode: FightMode = mode
        self.use_roboflow: bool = (
            _env_flag_default() if use_roboflow is None else use_roboflow
        )
        self._roboflow_disabled_reason: str | None = None

        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0, high=255, shape=PIXELS_SHAPE, dtype=np.uint8
                ),
                "troops": spaces.Box(
                    low=0, high=255, shape=TROOPS_HEATMAP_SHAPE, dtype=np.uint8
                ),
                "hand": spaces.Box(
                    low=0,
                    high=NUM_CARD_IDS - 1,
                    shape=(4,),
                    dtype=np.int32,
                ),
                "hand_affordable": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.uint8
                ),
                "elixir": spaces.Box(
                    low=0, high=10, shape=(1,), dtype=np.uint8
                ),
                # Per-detection structured channel: top-K troop/tower
                # detections, each with class id, normalized position,
                # side, and confidence. The custom feature extractor
                # in rl.policy embeds det_class through the same card
                # vocabulary as `hand`, so a Knight on the board and a
                # Knight in hand share representation.
                "det_class": spaces.Box(
                    low=0,
                    high=NUM_CARD_IDS - 1,
                    shape=(NUM_DETECTION_SLOTS,),
                    dtype=np.int32,
                ),
                "det_pos": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(NUM_DETECTION_SLOTS, 2),
                    dtype=np.float32,
                ),
                "det_side": spaces.Box(
                    low=0, high=NUM_SIDES - 1,
                    shape=(NUM_DETECTION_SLOTS,),
                    dtype=np.uint8,
                ),
                "det_conf": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(NUM_DETECTION_SLOTS,),
                    dtype=np.float32,
                ),
            }
        )
        self.reward_calc = RewardCalculator()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        decoded = decode(int(action))

        step_info = StepInfo(no_op=decoded.is_noop)
        if not decoded.is_noop:
            card_x, card_y = CARD_SLOTS[decoded.card]
            click(card_x, card_y)
            interruptible_sleep(CARD_SELECT_DELAY)
            click(decoded.x, decoded.y)
            step_info.card_played = True
            step_info.play_x = decoded.x
            step_info.play_y = decoded.y

        interruptible_sleep(FRAME_SKIP_SECONDS)

        # One screenshot powers detection, reward shaping, AND the
        # observation. Detection is the slowest of the three (~250ms),
        # so we run it before reward/obs to ensure a single round-trip.
        frame = get_screen()
        detections = self._maybe_detect(frame)
        reward, terminated = self.reward_calc.calculate(
            step_info, frame=frame, detections=detections
        )
        obs = self._build_obs(frame, detections)
        info: dict[str, Any] = {
            "card": decoded.card if not decoded.is_noop else -1,
            "x": decoded.x,
            "y": decoded.y,
            "hand": obs["hand"].tolist(),
            "hand_affordable": obs["hand_affordable"].tolist(),
            "elixir": int(obs["elixir"][0]),
            "n_detections": len(detections) if detections is not None else 0,
        }
        return obs, reward, terminated, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Self-healing episode reset.

        3 in-process attempts to return-to-main + start-battle. If all
        three fail, fall back to emulator.restart() — the same escape
        hatch pyclashbot's production state machine uses
        (pyclashbot/bot/states.py:257). This means a stuck-screen stall
        costs at most ~30 s of restart time instead of pausing the
        whole training run for human intervention.
        """
        super().reset(seed=seed)
        self.reward_calc.reset()

        for _ in range(3):
            if is_in_battle():
                break
            if return_to_main_menu() and start_battle(
                self.mode, start_timeout=RESET_WAIT_TIMEOUT
            ):
                break
            # Kick the emulator with back-key before retrying — most
            # commonly the bot is trapped in a popup whose visual
            # template we don't have yet.
            try:
                get_emulator().send_back_key()
            except Exception as e:
                print(f"[ClashRoyaleEnv.reset] back-key send failed: {e}")
            interruptible_sleep(2)
        else:
            print("[ClashRoyaleEnv.reset] 3 attempts failed; restarting emulator.")
            get_emulator().restart()
            return_to_main_menu()
            start_battle(self.mode, start_timeout=RESET_WAIT_TIMEOUT)

        frame = get_screen()
        # Skip detection at reset — the first frame is often the
        # "battle starting" countdown without any troops, so calling
        # Roboflow here mostly wastes ~250ms.
        return self._build_obs(frame, detections=None), {}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return get_screen()
        return None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _maybe_detect(self, frame: np.ndarray) -> list[TroopDetection] | None:
        """Run Roboflow detection if enabled, else return None.

        Detection failures (network, model, server) are caught and the
        feature is auto-disabled for the rest of the episode. The
        observation falls back to a zero heatmap and the reward
        calculator falls back to its HSV-based components, so a broken
        Roboflow setup degrades to "no troop signal" instead of
        crashing the training loop.
        """
        if not self.use_roboflow or self._roboflow_disabled_reason:
            return None
        try:
            return detect_troops(frame)
        except Exception as e:
            self._roboflow_disabled_reason = str(e)
            print(
                "[ClashRoyaleEnv] Roboflow detection failed; disabling for this "
                f"episode. Cause: {type(e).__name__}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _build_obs(
        self,
        frame: np.ndarray,
        detections: list[TroopDetection] | None,
    ) -> dict[str, np.ndarray]:
        """Compose the structured observation from a single frame.

        Card classification + elixir reading happen off the same frame
        as the pixel downscale, so all signals are temporally aligned
        (no risk of the "hand" reading being one step ahead of
        "pixels"). The card classifier runs on the full-resolution
        image because its thresholds are tuned to 419x633.
        """
        pixels = self._preprocess_pixels(frame)
        troops = rasterize_detections(detections)

        try:
            card_ids, affordable = read_hand(frame)
        except Exception:
            # Pre-battle frames / menu frames don't have cards — return
            # UNKNOWNs. The observation space guarantees 4 entries so
            # the policy always sees a fixed-shape input.
            card_ids, affordable = [0, 0, 0, 0], [False, False, False, False]

        try:
            elixir = read_elixir(frame)
        except Exception:
            elixir = 0

        # Pass frame so encode_detections can sample HP bars for
        # team classification — far more reliable than y-fallback when
        # troops cross the river.
        det_class, det_pos, det_side, det_conf = encode_detections(
            detections, frame=frame
        )

        return {
            "pixels": pixels,
            "troops": troops,
            "hand": np.asarray(card_ids, dtype=np.int32),
            "hand_affordable": np.asarray(
                [int(a) for a in affordable], dtype=np.uint8
            ),
            "elixir": np.asarray([elixir], dtype=np.uint8),
            "det_class": det_class,
            "det_pos": det_pos,
            "det_side": det_side,
            "det_conf": det_conf,
        }

    @staticmethod
    def _preprocess_pixels(screen: np.ndarray) -> np.ndarray:
        if screen.shape[0] != SCREEN_H or screen.shape[1] != SCREEN_W:
            screen = cv2.resize(screen, (SCREEN_W, SCREEN_H))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]
