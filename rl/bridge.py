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

from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from pyclashbot.emulators.base import BaseEmulatorController

FightMode = Literal["Classic 1v1", "Classic 2v2", "Trophy Road"]

SCREEN_W = 419
SCREEN_H = 633

_emulator: BaseEmulatorController | None = None
_logger: Logger | None = None


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
    emulator_type: Literal["memu", "bluestacks"] = "memu",
    render_mode: str = "directx",
    debug_mode: bool = False,
) -> BaseEmulatorController:
    """Lazily construct and cache the emulator controller.

    debug_mode=True skips configure/restart so you can attach to an already
    running VM for iterative development. Flip it to False when you want
    the full supervised boot + Clash startup path.
    """
    global _emulator
    if _emulator is not None:
        return _emulator

    logger = get_logger()
    if emulator_type == "memu":
        from pyclashbot.emulators.memu import MemuEmulatorController

        _emulator = MemuEmulatorController(
            logger, render_mode=render_mode, debug_mode=debug_mode
        )
    elif emulator_type == "bluestacks":
        from pyclashbot.emulators.bluestacks import BlueStacksEmulatorController

        _emulator = BlueStacksEmulatorController(logger, render_mode=render_mode)  # type: ignore[call-arg]
    else:
        raise ValueError(f"Unknown emulator type: {emulator_type}")
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
