"""Action-space encoding for the Clash Royale RL environment.

Geometry is calibrated against the fixed 419x633 screen that
pyclashbot's MEMU_CONFIGURATION enforces:

    pyclashbot/emulators/memu.py::MEMU_CONFIGURATION
        resolution_width  = 419
        resolution_height = 633

Card slot pixel positions mirror HAND_CARDS_COORDS in pyclashbot/bot/fight.py,
so selecting a card here uses the same pixels the scripted bot does.

The arena bounds below cover the AGENT'S half of the arena — the
lower region between the river (~y=320) and the hand row (~y=555).
Adjust ARENA after eyeballing a real screenshot.
"""
from __future__ import annotations

from dataclasses import dataclass

GRID_W = 10
GRID_H = 10
NUM_CARDS = 4

# Confirmed from pyclashbot/bot/fight.py::HAND_CARDS_COORDS.
CARD_SLOTS: list[tuple[int, int]] = [
    (142, 561),
    (210, 563),
    (272, 561),
    (341, 563),
]

# Full playable arena in pixel space. x spans the board minus UI margins;
# y spans from just past the opp king tower down to just above the card
# tray, covering BOTH halves of the arena.
#
# Placements in the enemy half (y < ~300) are normally rejected by the
# game, costing the agent a click without spending elixir — that's what
# INVALID_PLAY_PENALTY in reward.py is for. Once an opp princess tower
# is destroyed, the enemy half of THAT lane becomes legal, and the
# LANE_OPENED_BONUS in reward.py rewards placements there. The action
# space stays fixed (4 cards x 10 x 10 grid + no-op = 401) so loading
# an existing PPO model still works; the per-cell pixel mapping just
# covers more of the board.
ARENA: dict[str, int] = {
    "x_min": 30,
    "x_max": 389,
    "y_min": 80,
    "y_max": 520,
}

NO_OP: int = NUM_CARDS * GRID_W * GRID_H  # last action index = wait
TOTAL_ACTIONS: int = NO_OP + 1  # 401


@dataclass(frozen=True)
class DecodedAction:
    card: int
    x: int
    y: int
    is_noop: bool


def decode(action: int) -> DecodedAction:
    """Map a flat action index to (card_idx, pixel_x, pixel_y)."""
    if not 0 <= action < TOTAL_ACTIONS:
        raise ValueError(f"Action {action} out of range [0, {TOTAL_ACTIONS})")

    if action == NO_OP:
        return DecodedAction(card=0, x=0, y=0, is_noop=True)

    card = action // (GRID_W * GRID_H)
    remainder = action % (GRID_W * GRID_H)
    gx = remainder % GRID_W
    gy = remainder // GRID_W

    # gx/gy are in [0, GRID_W-1] / [0, GRID_H-1]; map to pixels linearly.
    x_span = ARENA["x_max"] - ARENA["x_min"]
    y_span = ARENA["y_max"] - ARENA["y_min"]
    px = int(ARENA["x_min"] + gx * x_span / (GRID_W - 1))
    py = int(ARENA["y_min"] + gy * y_span / (GRID_H - 1))
    return DecodedAction(card=card, x=px, y=py, is_noop=False)


def encode(card: int, gx: int, gy: int) -> int:
    """Inverse of decode() for grid coordinates. Useful for expert demos/tests."""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card {card} out of range")
    if not 0 <= gx < GRID_W or not 0 <= gy < GRID_H:
        raise ValueError(f"grid ({gx}, {gy}) out of range")
    return card * (GRID_W * GRID_H) + gy * GRID_W + gx
