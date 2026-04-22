"""Gymnasium environment wrapping a live Clash Royale emulator.

Runs in lockstep with pyclashbot: each step taps up to two pixels on the
emulator, waits FRAME_SKIP_SECONDS for the game to evolve, and then
returns a downsampled grayscale frame.

Timing notes:
    * The game runs in real time. FRAME_SKIP_SECONDS controls how much
      wall-clock the environment gives the game between decisions.
    * pyclashbot bans `time.sleep` in favour of `interruptible_sleep`
      (see pyproject.toml's TID251 config) so the agent thread can be
      cancelled cleanly from the GUI.
"""
from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pyclashbot.utils.cancellation import interruptible_sleep
from rl.action_map import CARD_SLOTS, TOTAL_ACTIONS, decode
from rl.bridge import (
    SCREEN_H,
    SCREEN_W,
    FightMode,
    click,
    get_screen,
    is_in_battle,
    return_to_main_menu,
    start_battle,
)
from rl.reward import RewardCalculator, StepInfo

OBS_SIZE = 84
OBS_SHAPE = (OBS_SIZE, OBS_SIZE, 1)

# Real seconds between agent decisions. Keep this >= 0.3s so the emulator
# has time to render the consequences of a card placement.
FRAME_SKIP_SECONDS = 0.6
CARD_SELECT_DELAY = 0.08
RESET_WAIT_TIMEOUT = 120


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, mode: FightMode = "Classic 1v1") -> None:
        super().__init__()
        self.mode: FightMode = mode
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8
        )
        self.reward_calc = RewardCalculator()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        decoded = decode(int(action))

        step_info = StepInfo(no_op=decoded.is_noop)
        if not decoded.is_noop:
            card_x, card_y = CARD_SLOTS[decoded.card]
            click(card_x, card_y)
            interruptible_sleep(CARD_SELECT_DELAY)
            click(decoded.x, decoded.y)
            # We don't verify the card was actually playable (elixir,
            # hand contents). Treat the attempt as a "card_played"
            # signal; cleanup is the agent's job to learn.
            step_info.card_played = True

        interruptible_sleep(FRAME_SKIP_SECONDS)

        reward, terminated = self.reward_calc.calculate(step_info)
        obs = self._preprocess(get_screen())
        info: dict[str, Any] = {
            "card": decoded.card if not decoded.is_noop else -1,
            "x": decoded.x,
            "y": decoded.y,
        }
        return obs, reward, terminated, False, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.reward_calc.reset()

        if not is_in_battle():
            return_to_main_menu()
            start_battle(self.mode, start_timeout=RESET_WAIT_TIMEOUT)

        return self._preprocess(get_screen()), {}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return get_screen()
        return None

    @staticmethod
    def _preprocess(screen: np.ndarray) -> np.ndarray:
        if screen.shape[0] != SCREEN_H or screen.shape[1] != SCREEN_W:
            # defensive: upstream guarantees 419x633 but we don't crash if not
            screen = cv2.resize(screen, (SCREEN_W, SCREEN_H))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]
