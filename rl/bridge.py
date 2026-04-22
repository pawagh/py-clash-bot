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

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pyclashbot.bot.fight import (
    HAND_CARDS_COORDS,
    get_to_main_after_fight,
    start_fight,
)
from pyclashbot.bot.nav import (
    check_if_battle_has_ended,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    select_mode,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.utils.logger import Logger
from pyclashbot.utils.platform import is_macos

if TYPE_CHECKING:
    from pyclashbot.emulators.base import BaseEmulatorController

FightMode = Literal["Classic 1v1", "Classic 2v2", "Trophy Road"]
EmulatorType = Literal["bluestacks", "memu"]

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


def start_battle(mode: FightMode = "Classic 1v1", start_timeout: int = 120) -> bool:
    """Navigate from the main menu to an active battle.

    Mirrors what fight.do_fight_state does on entry:
      1. ensure we're on main,
      2. select_mode(...) to pick the fight type,
      3. start_fight(...) to click the battle button,
      4. wait_for_battle_start(...) for the scoreboard pixels.
    """
    emulator = get_emulator()
    logger = get_logger()

    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        return False
    if not select_mode(emulator, mode):
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
    "HAND_CARDS_COORDS",
    "SCREEN_H",
    "SCREEN_W",
    "click",
    "get_emulator",
    "get_logger",
    "get_screen",
    "is_battle_over",
    "is_in_battle",
    "is_on_main_menu",
    "return_to_main_menu",
    "set_emulator",
    "start_battle",
    "swipe",
]
