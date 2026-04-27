"""Post-battle recovery: get the bot from any post-fight screen back to
the Trophy Road main menu so the next RL episode can start cleanly.

The legacy `get_to_main_after_fight` only handled trophy-reward popups
and the OK/Exit button — anything else (Festival Market, Welcome Gift,
Community Event, Battle Deck screen, Merge Tactics tab, ...) caused the
RL training loop to stall on a screen and need manual rescue.

This module adds:

  is_on_trophy_road_battle_screen(emulator) -> bool
      Positive confirmation. Only when this returns True is it safe for
      the RL bridge to click the Battle button.

  detect_bad_screen(emulator) -> str | None
      Negative confirmation. Returns the name of a known bad screen
      (e.g. "festival_market") so logs name the screen the bot drifted
      onto. Recovery is the same regardless of which bad screen — this
      is purely diagnostic so the user knows which template to add next.

  try_dismiss_current_screen(emulator, logger) -> bool
      Tries the *visual* dismissals in priority order: trophy reward OK,
      trophy-road overlay OK, post-battle OK/Exit, popup close-X, popup
      bottom-Close. Returns True iff something was clicked.

The escalation ladder above the visual layer (back-key, home-tab click)
lives in `get_to_main_after_fight` itself.
"""
from __future__ import annotations

import os
from os.path import abspath, dirname, join

from pyclashbot.bot.nav import (
    check_for_trophy_reward_menu,
    check_if_battle_mode_is_selected,
    check_if_on_clash_main_menu,
    handle_trophy_reward_menu,
)
from pyclashbot.detection.image_rec import find_image
from pyclashbot.utils.logger import Logger

# Folders the dismissal layer probes, in priority order.
# Each entry: (folder_name, tolerance, label_for_logs).
# Tolerances were measured against the supplied reference screenshots:
# trophy_road OK and welcome-gift X are visually generic enough that
# loose tolerances false-match on other screens, so they're tightened.
_DISMISSAL_TEMPLATES: list[tuple[str, float, str]] = [
    ("trophy_road_overlay_ok", 0.95, "trophy_road_overlay_ok"),
    ("popup_close_x", 0.88, "popup_close_x"),
    ("popup_bottom_close", 0.88, "popup_bottom_close"),
]

# Bad-screen identifiers. Each entry: (folder_name, tolerance).
# welcome_gift's title band is similar enough to other reward popups
# that 0.85 produces false positives — bumped to 0.92 per measurement.
_BAD_SCREEN_FOLDERS: list[tuple[str, float]] = [
    ("bad_screen_festival_market", 0.88),
    ("bad_screen_festival_market_boost", 0.88),
    ("bad_screen_community_event", 0.88),
    ("bad_screen_welcome_gift", 0.92),
]

_REF_ROOT = abspath(join(dirname(__file__), "..", "detection", "reference_images"))


def _folder_has_templates(folder: str) -> bool:
    """find_image crashes on an empty folder (ThreadPoolExecutor max_workers=0).

    Guard so dropping in a new folder before populating it never breaks
    the recovery loop.
    """
    path = join(_REF_ROOT, folder)
    if not os.path.isdir(path):
        return False
    return any(name.endswith((".png", ".jpg")) for name in os.listdir(path))


def is_on_trophy_road_battle_screen(emulator) -> bool:
    """True iff we are on the main menu AND Trophy Road is the selected mode.

    This is the only state from which the RL bridge is allowed to click
    the Battle button — clicking it on any other screen risks starting
    a wrong-mode match (Merge Tactics, etc).
    """
    if not check_if_on_clash_main_menu(emulator):
        return False
    return bool(check_if_battle_mode_is_selected(emulator, "Trophy Road"))


def detect_bad_screen(emulator) -> str | None:
    """Return the name of a known bad screen if matched, else None.

    Pure diagnostic — the dismissal action is the same regardless of
    which bad screen matched. Logging the name tells the user which
    `bad_screen_*` folder is paying off and which one to add next.
    """
    iar = emulator.screenshot()
    if iar is None:
        return None
    for folder, tol in _BAD_SCREEN_FOLDERS:
        if not _folder_has_templates(folder):
            continue
        if find_image(iar, folder, tolerance=tol) is not None:
            return folder.replace("bad_screen_", "")
    return None


def try_dismiss_current_screen(emulator, logger: Logger) -> str | None:
    """Try every known dismissal action; return the label of what we clicked.

    Priority order matches what's most likely to be on screen right after
    a battle:
      1. Trophy reward menu (existing pixel check).
      2. Trophy-road overlay OK button (the ladder graphic).
      3. Generic popup close-X (top-right of reward popups).
      4. Generic popup bottom-Close (Festival Market, etc).

    Returns the dismissed label (for logs) or None if nothing matched.
    """
    # 1. Existing trophy-reward menu (cheap pixel check, no template).
    if check_for_trophy_reward_menu(emulator):
        handle_trophy_reward_menu(emulator, logger, printmode=False)
        return "trophy_reward_menu"

    iar = emulator.screenshot()
    if iar is None:
        return None

    # 2-4. Visual templates.
    for folder, tol, label in _DISMISSAL_TEMPLATES:
        if not _folder_has_templates(folder):
            continue
        coord = find_image(iar, folder, tolerance=tol)
        if coord is not None:
            emulator.click(coord[0], coord[1])
            return label

    return None
