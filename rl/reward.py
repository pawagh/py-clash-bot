"""Reward shaping for the Clash Royale RL environment.

This first pass intentionally starts simple so we can smoke-test the full
PPO pipeline before investing in heavy CV work:

    * tiny living cost on every step (discourages pure no-op spam),
    * large terminal bonus when the battle ends,
    * optional tiny card-played bonus (set to 0.0 if you don't want it).

Crown/HP reading is marked TODO below. When you're ready to shape rewards
densely, implement _read_crowns() with template matching against the crown
UI or with ELIXIR_COORDS-style pixel sampling. The existing helpers in
pyclashbot.bot.fight.check_pixels_for_win_in_battle_log illustrate the
pattern; the same approach works for the mid-battle crown counter that
appears over each tower.
"""
from __future__ import annotations

from dataclasses import dataclass

from rl.bridge import is_battle_over, is_in_battle


@dataclass
class StepInfo:
    """Lightweight side-channel the env passes into RewardCalculator."""

    card_played: bool = False
    no_op: bool = False


class RewardCalculator:
    """Tracks per-episode deltas and emits a scalar reward per step."""

    # Coefficients — tune these once you have a working training loop.
    STEP_COST: float = -0.005
    NOOP_COST: float = -0.002
    CARD_PLAYED_BONUS: float = 0.01
    WIN_BONUS: float = 5.0
    LOSS_PENALTY: float = -5.0
    # Per-crown shaping (requires _read_crowns to be implemented).
    CROWN_REWARD: float = 2.0

    def __init__(self) -> None:
        self.prev_my_crowns = 0
        self.prev_opp_crowns = 0
        self._was_in_battle = False

    def reset(self) -> None:
        self.prev_my_crowns = 0
        self.prev_opp_crowns = 0
        self._was_in_battle = False

    def calculate(self, info: StepInfo) -> tuple[float, bool]:
        """Return (reward, terminated) for the current step.

        Terminated is True only on the first frame where the battle has
        ended — this keeps Gymnasium semantics clean.
        """
        reward = self.STEP_COST
        if info.no_op:
            reward += self.NOOP_COST
        if info.card_played:
            reward += self.CARD_PLAYED_BONUS

        my_crowns, opp_crowns = self._read_crowns()
        if my_crowns > self.prev_my_crowns:
            reward += self.CROWN_REWARD * (my_crowns - self.prev_my_crowns)
        if opp_crowns > self.prev_opp_crowns:
            reward -= self.CROWN_REWARD * (opp_crowns - self.prev_opp_crowns)
        self.prev_my_crowns = my_crowns
        self.prev_opp_crowns = opp_crowns

        currently_in_battle = is_in_battle()
        terminated = False
        if self._was_in_battle and not currently_in_battle and is_battle_over():
            terminated = True
            reward += self._terminal_reward(my_crowns, opp_crowns)
        self._was_in_battle = self._was_in_battle or currently_in_battle

        return reward, terminated

    def _terminal_reward(self, my_crowns: int, opp_crowns: int) -> float:
        if my_crowns > opp_crowns:
            return self.WIN_BONUS
        if my_crowns < opp_crowns:
            return self.LOSS_PENALTY
        return 0.0  # draw

    def _read_crowns(self) -> tuple[int, int]:
        """Return (my_crowns, opp_crowns) in [0, 3].

        TODO: replace this stub with real CV. Candidate approaches:
          * template match against cropped crown icons over each tower,
          * sample a handful of known pixel positions (like ELIXIR_COORDS
            in pyclashbot/bot/fight.py) and count gold-tinted hits.

        Returning zeros keeps the CROWN_REWARD term dormant so the rest
        of the reward signal works on day one.
        """
        return 0, 0
