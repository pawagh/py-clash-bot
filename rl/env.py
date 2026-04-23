"""Gymnasium environment wrapping a live Clash Royale emulator.

Runs in lockstep with pyclashbot: each step taps up to two pixels on the
emulator, waits FRAME_SKIP_SECONDS for the game to evolve, and then
returns a structured observation (downsampled grayscale frame + hand /
elixir state).

Observation layout (Dict):
    pixels           uint8  (OBS_SIZE, OBS_SIZE, 1)   arena screenshot
    hand             int32  (4,)                       pyclashbot card ids
    hand_affordable  uint8  (4,)                       0/1 per slot
    elixir           uint8  (1,)                       current 0..10

Every field is a gym.Box so VecFrameStack can stack all of them across
n_stack frames. Stacking the structured fields is intentional: it gives
the policy an implicit cycle history (what cards rotated out) and the
elixir-regen trajectory, for free.

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
    DEFAULT_MODE,
    NUM_CARD_IDS,
    SCREEN_H,
    SCREEN_W,
    FightMode,
    click,
    get_screen,
    is_in_battle,
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


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, mode: FightMode = DEFAULT_MODE) -> None:
        super().__init__()
        self.mode: FightMode = mode
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0, high=255, shape=PIXELS_SHAPE, dtype=np.uint8
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

        interruptible_sleep(FRAME_SKIP_SECONDS)

        # One screenshot powers reward shaping AND the observation.
        frame = get_screen()
        reward, terminated = self.reward_calc.calculate(step_info, frame=frame)
        obs = self._build_obs(frame)
        info: dict[str, Any] = {
            "card": decoded.card if not decoded.is_noop else -1,
            "x": decoded.x,
            "y": decoded.y,
            "hand": obs["hand"].tolist(),
            "hand_affordable": obs["hand_affordable"].tolist(),
            "elixir": int(obs["elixir"][0]),
        }
        return obs, reward, terminated, False, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.reward_calc.reset()

        if not is_in_battle():
            return_to_main_menu()
            start_battle(self.mode, start_timeout=RESET_WAIT_TIMEOUT)

        return self._build_obs(get_screen()), {}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return get_screen()
        return None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _build_obs(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Compose the structured observation from a single frame.

        Card classification + elixir reading happen off the same frame
        as the pixel downscale, so all signals are temporally aligned
        (no risk of the "hand" reading being one step ahead of
        "pixels"). The card classifier runs on the full-resolution
        image because its thresholds are tuned to 419x633.
        """
        pixels = self._preprocess_pixels(frame)

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

        return {
            "pixels": pixels,
            "hand": np.asarray(card_ids, dtype=np.int32),
            "hand_affordable": np.asarray(
                [int(a) for a in affordable], dtype=np.uint8
            ),
            "elixir": np.asarray([elixir], dtype=np.uint8),
        }

    @staticmethod
    def _preprocess_pixels(screen: np.ndarray) -> np.ndarray:
        if screen.shape[0] != SCREEN_H or screen.shape[1] != SCREEN_W:
            screen = cv2.resize(screen, (SCREEN_W, SCREEN_H))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]
