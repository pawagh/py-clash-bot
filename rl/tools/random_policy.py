"""Random-policy baseline runner.

Before running training, this is the single most useful integration test.

From the repo root, with BlueStacks open and Clash Royale on the main menu:

    uv run python -m rl.tools.random_policy --episodes 2

What it does:
    1. Boots the emulator via rl.bridge (no-op if already booted).
    2. Plays N full matches with a uniformly random policy.
    3. Prints per-episode length, reward, and per-step wall time.

What to check in the output:
    * Episodes actually start and end without the env hanging.
    * Per-step wall time ≈ FRAME_SKIP_SECONDS (rl.env). If it's much
      larger, screenshots are slow and training will be unbearable.
    * Mean episode length roughly matches real match length divided by
      FRAME_SKIP_SECONDS (3 min / 0.6s ≈ 300 steps).
    * All rewards are finite — NaN means reward shaping is broken.

This output becomes your baseline. A trained PPO agent should beat the
mean episode reward here within a few thousand timesteps; if it doesn't,
something is wrong with the reward signal or the env.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from pyclashbot.utils.platform import is_macos
from rl.bridge import get_emulator
from rl.env import FRAME_SKIP_SECONDS, ClashRoyaleEnv


def _boot_emulator() -> None:
    if is_macos():
        get_emulator(
            emulator_type="bluestacks",
            render_settings={"graphics_renderer": "vlcn"},
        )
    else:
        get_emulator(
            emulator_type="memu", render_mode="directx", debug_mode=True
        )


def run(episodes: int, max_steps_per_episode: int, seed: int) -> int:
    print("Booting emulator (first run may take 30-60s)...")
    _boot_emulator()

    rng = np.random.default_rng(seed)
    env = ClashRoyaleEnv()

    all_rewards: list[float] = []
    all_lengths: list[int] = []
    all_step_times: list[float] = []

    for ep in range(episodes):
        print(f"\n=== Episode {ep + 1}/{episodes} ===")
        print("  resetting (navigating to battle)...")
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

        episode_reward = 0.0
        episode_length = 0
        terminated = False
        for step_idx in range(max_steps_per_episode):
            action = int(rng.integers(0, env.action_space.n))
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            dt = time.perf_counter() - t0
            all_step_times.append(dt)
            episode_reward += reward
            episode_length += 1
            if step_idx % 50 == 0:
                print(
                    f"  step {step_idx:4d} action={action:3d} "
                    f"reward={reward:+.3f} total={episode_reward:+.3f}"
                )
            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        status = "terminated" if terminated else "truncated (hit max steps)"
        print(
            f"  {status} after {episode_length} steps, reward={episode_reward:+.3f}"
        )

    if all_rewards:
        print("\n=== Summary ===")
        print(f"  episodes:       {len(all_rewards)}")
        print(
            f"  episode reward: mean={statistics.mean(all_rewards):+.3f}  "
            f"stdev={statistics.pstdev(all_rewards):.3f}"
        )
        print(
            f"  episode length: mean={statistics.mean(all_lengths):.1f}  "
            f"stdev={statistics.pstdev(all_lengths):.1f}"
        )
        if all_step_times:
            print(
                f"  step time:      mean={statistics.mean(all_step_times):.3f}s  "
                f"(FRAME_SKIP_SECONDS={FRAME_SKIP_SECONDS})"
            )
        if any(not np.isfinite(r) for r in all_rewards):
            print("  [!] Non-finite reward detected — reward shaping is broken")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return run(args.episodes, args.max_steps, args.seed)


if __name__ == "__main__":
    sys.exit(main())
