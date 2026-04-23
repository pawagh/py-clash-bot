"""Visual calibration aid — renders card slots and arena grid onto a screenshot.

Run from the repo root with BlueStacks open (a battle screen is best so you
can see the card tray):

    uv run python -m rl.tools.calibrate

Writes calibration.png next to the repo root. Open it and confirm:

    * the red circles sit exactly on the centers of the four card slots,
    * the green rectangle covers YOUR half of the arena only,
    * the green grid dots fall on playable tiles, not on HUD elements
      (crowns, elixir bar, emote button at (67, 521)).

If anything is off, edit rl/action_map.py::ARENA or ::CARD_SLOTS and rerun
until the overlay looks right. Bad coords here == the policy clicks random
UI elements forever and never learns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from pyclashbot.utils.platform import is_macos
from rl.action_map import ARENA, CARD_SLOTS, GRID_H, GRID_W, decode, encode
from rl.bridge import SCREEN_H, SCREEN_W, get_emulator, get_screen

OUT = Path("calibration.png")


def overlay(frame: np.ndarray) -> np.ndarray:
    canvas = frame.copy()

    # Arena bounding box — where the agent is allowed to place cards.
    cv2.rectangle(
        canvas,
        (ARENA["x_min"], ARENA["y_min"]),
        (ARENA["x_max"], ARENA["y_max"]),
        color=(0, 255, 0),
        thickness=1,
    )

    # One dot per (gx, gy) — these are the exact pixels decode() emits.
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            d = decode(encode(0, gx, gy))
            cv2.circle(canvas, (d.x, d.y), radius=2, color=(0, 255, 0), thickness=-1)

    # Card slots, labelled so you can tell which is which.
    for i, (x, y) in enumerate(CARD_SLOTS):
        cv2.circle(canvas, (x, y), radius=8, color=(0, 0, 255), thickness=2)
        cv2.putText(
            canvas,
            str(i),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    h, w = canvas.shape[:2]
    cv2.putText(
        canvas,
        f"{w}x{h}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _boot_emulator() -> None:
    """Eagerly construct the controller so lazy errors surface here, not later."""
    if is_macos():
        get_emulator(
            emulator_type="bluestacks",
            render_settings={"graphics_renderer": "vlcn"},
        )
    else:
        get_emulator(
            emulator_type="memu", render_mode="directx", debug_mode=True
        )


def main() -> int:
    print("[1/3] Booting emulator controller (first run may take 30-60s)...")
    _boot_emulator()

    print("[2/3] Taking screenshot...")
    frame = get_screen()
    if frame.shape[:2] != (SCREEN_H, SCREEN_W):
        print(
            f"      WARNING: got {frame.shape[:2]}, expected ({SCREEN_H}, {SCREEN_W}). "
            f"Is the emulator running at 419x633?"
        )

    print("[3/3] Overlaying card slots and arena grid...")
    canvas = overlay(frame)
    cv2.imwrite(str(OUT), canvas)
    print(f"      wrote {OUT.resolve()}")
    print()
    print("Open the PNG and confirm:")
    print("  * red circles (0-3) sit on the four card slot centers,")
    print("  * green rectangle covers your half of the arena,")
    print("  * green dots don't land on HUD elements.")
    print("If anything is off, edit rl/action_map.py and rerun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
