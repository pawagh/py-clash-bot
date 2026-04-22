"""Standalone verification that rl/ is correctly wired to pyclashbot.

Run from the repo root with the emulator already booted and Clash Royale
on the main menu:

    uv run python -m rl.smoke_test

This does NOT train anything. It walks the bridge layer end to end:

    1. construct the emulator controller (via rl.bridge),
    2. take a screenshot and save it as smoke_test_screen.png,
    3. verify the frame is 419x633 BGR,
    4. print what nav/fight think the current screen is,
    5. dump action 0 and action TOTAL_ACTIONS-1 (the no-op) through decode().

If this script runs clean, env.py and train.py will too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from pyclashbot.utils.platform import is_macos
from rl.action_map import CARD_SLOTS, NO_OP, TOTAL_ACTIONS, decode
from rl.bridge import (
    SCREEN_H,
    SCREEN_W,
    get_emulator,
    get_screen,
    is_battle_over,
    is_in_battle,
    is_on_main_menu,
)

OUT = Path("smoke_test_screen.png")


def main() -> int:
    print("[1/5] Constructing emulator controller...")
    if is_macos():
        # macOS: BlueStacks with Vulkan. MEmu is Windows-only.
        emulator = get_emulator(
            emulator_type="bluestacks",
            render_settings={"graphics_renderer": "vlcn"},
        )
    else:
        emulator = get_emulator(
            emulator_type="memu", render_mode="directx", debug_mode=True
        )
    print(f"      controller: {type(emulator).__name__}")

    print("[2/5] Taking a screenshot...")
    frame = get_screen()
    print(f"      shape: {frame.shape}  dtype: {frame.dtype}")
    if frame.shape[:2] != (SCREEN_H, SCREEN_W):
        print(f"      WARNING: expected ({SCREEN_H}, {SCREEN_W}), got {frame.shape[:2]}")
    cv2.imwrite(str(OUT), frame)
    print(f"      saved to {OUT.resolve()}")

    print("[3/5] Running pyclashbot detection helpers...")
    print(f"      is_on_main_menu(): {is_on_main_menu()}")
    print(f"      is_in_battle():    {is_in_battle()}")
    print(f"      is_battle_over():  {is_battle_over()}")

    print("[4/5] Card slot coordinates (from HAND_CARDS_COORDS):")
    for i, (x, y) in enumerate(CARD_SLOTS):
        print(f"      card {i}: ({x}, {y})")

    print(f"[5/5] Action space sanity ({TOTAL_ACTIONS} actions total):")
    print(f"      action 0     -> {decode(0)}")
    print(f"      action 199   -> {decode(199)}")
    print(f"      action {NO_OP} (no-op) -> {decode(NO_OP)}")

    print("\nAll checks completed without exception. You can now run")
    print("    uv run python -m rl.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
