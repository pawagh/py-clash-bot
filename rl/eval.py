"""Evaluate a trained checkpoint by playing N battles deterministically.

Reports per-battle outcome (W / L / D) and overall win rate. Outcome is
inferred from the terminal reward sign — RewardCalculator returns
WIN_BONUS (+5.0) on wins, LOSS_PENALTY (-5.0) on losses, 0 on draws.
We use a magnitude threshold of 2.5 so a battle with strong shaped
rewards but a draw outcome can't be miscounted as a win/loss.

Usage:

    # Auto-pick the latest checkpoint:
    uv run python -m rl.eval --episodes 20

    # Specific checkpoint:
    uv run python -m rl.eval --checkpoint ./checkpoints/clash_ppo_50000_steps.zip

    # Stochastic policy (use exploration distribution rather than argmax):
    uv run python -m rl.eval --episodes 20 --stochastic

The env wraps DummyVecEnv + VecFrameStack(n_stack=8) just like training,
so the loaded model sees observations in the same shape it was trained
on. No code change needed in env.py / train.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from rl.env import ClashRoyaleEnv

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_PREFIX = "clash_ppo"
# Magnitude that distinguishes a terminal win/loss reward from
# accumulated shaping. WIN_BONUS=5, LOSS_PENALTY=-5, shaping ≪ 1 per
# step so a 2.5 threshold is conservative but safe.
WIN_LOSS_THRESHOLD = 2.5


def _find_latest_checkpoint() -> Path | None:
    """Return the newest auto-saved checkpoint, or None."""
    candidates = list(CHECKPOINT_DIR.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"))
    final = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_final.zip"
    if final.exists():
        candidates.append(final)
    if not candidates:
        return None

    def step_of(p: Path) -> int:
        if p.stem.endswith("final"):
            return 10**12  # always rank highest
        try:
            return int(p.stem.split("_")[-2])
        except (ValueError, IndexError):
            return -1

    return max(candidates, key=step_of)


def _classify_outcome(last_reward: float) -> str:
    if last_reward > WIN_LOSS_THRESHOLD:
        return "W"
    if last_reward < -WIN_LOSS_THRESHOLD:
        return "L"
    return "D"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Specific checkpoint zip. Defaults to latest in ./checkpoints/.")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of battles to play (default: 20).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from the policy distribution (default: argmax / deterministic).")
    args = p.parse_args()

    ckpt = args.checkpoint or _find_latest_checkpoint()
    if ckpt is None or not ckpt.exists():
        print("ERROR: no checkpoint found. Pass --checkpoint <path> or "
              "ensure ./checkpoints/clash_ppo_*_steps.zip exists.")
        return 1

    print(f"Loading checkpoint: {ckpt}")
    print(f"Mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"Episodes: {args.episodes}\n")

    env = VecFrameStack(DummyVecEnv([ClashRoyaleEnv]), n_stack=8)
    model = PPO.load(str(ckpt), env=env)

    outcomes: list[str] = []
    rewards_total: list[float] = []
    durations_s: list[float] = []

    for ep in range(args.episodes):
        ep_start = time.time()
        obs = env.reset()
        ep_reward = 0.0
        last_reward = 0.0
        steps = 0
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, dones, _ = env.step(action)
            r = float(reward[0])
            ep_reward += r
            last_reward = r
            steps += 1
            terminated = bool(dones[0])

        outcome = _classify_outcome(last_reward)
        duration_s = time.time() - ep_start
        outcomes.append(outcome)
        rewards_total.append(ep_reward)
        durations_s.append(duration_s)

        running_w = outcomes.count("W")
        running_l = outcomes.count("L")
        running_d = outcomes.count("D")
        print(
            f"  Episode {ep + 1:>2d}/{args.episodes}: {outcome}  "
            f"steps={steps:3d}  ep_reward={ep_reward:+6.2f}  "
            f"terminal={last_reward:+6.2f}  "
            f"wall={duration_s:.0f}s  "
            f"running W/L/D = {running_w}/{running_l}/{running_d}",
            flush=True,
        )

    n = len(outcomes)
    wins = outcomes.count("W")
    losses = outcomes.count("L")
    draws = outcomes.count("D")
    win_rate = wins / n if n else 0.0
    avg_reward = sum(rewards_total) / n if n else 0.0
    avg_duration = sum(durations_s) / n if n else 0.0

    print("\n=== Final results ===")
    print(f"  Battles:     {n}")
    print(f"  Wins:        {wins}")
    print(f"  Losses:      {losses}")
    print(f"  Draws:       {draws}")
    print(f"  Win rate:    {win_rate * 100:.1f}%")
    print(f"  Avg reward:  {avg_reward:+.2f}")
    print(f"  Avg duration:{avg_duration:.0f}s/battle")
    print(f"  Total wall:  {sum(durations_s):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
