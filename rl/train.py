"""PPO training entry point.

Run from the repo root:

    uv run python -m rl.train                 # fresh run (or resume if a
                                              # checkpoint exists)
    uv run python -m rl.train --fresh         # force a fresh run, ignore
                                              # existing checkpoints
    uv run python -m rl.train --timesteps 25000
                                              # custom total budget
    uv run python -m rl.train --resume-from ./checkpoints/clash_ppo_10000_steps.zip
                                              # resume from a specific file
    uv run python -m rl.train --no-check      # skip the SB3 env sanity
                                              # check (saves ~30s of live
                                              # clicks on startup)

Resume semantics:
    * If a checkpoint exists in CHECKPOINT_DIR and --fresh is NOT set, the
      newest *_steps.zip is loaded and training picks up where it left
      off, preserving the global step counter, Adam optimizer state, and
      TensorBoard run name.
    * reset_num_timesteps=False is used so `model.num_timesteps` stays
      monotonic across runs. Stopping at 10h and resuming for another 10h
      is therefore equivalent to 20h continuous (modulo the at-most one
      on-policy rollout of experience lost when you ctrl-C mid-rollout).

The env drives a real emulator, so we use DummyVecEnv(1) + VecFrameStack.
Don't bump to SubprocVecEnv without first confirming your host can run N
BlueStacks/MEmu instances in parallel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from rl.env import ClashRoyaleEnv

CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./runs")
CHECKPOINT_PREFIX = "clash_ppo"
TB_LOG_NAME = "clash_ppo"
DEFAULT_TIMESTEPS = 50_000
SAVE_FREQ = 2_500


def build_env() -> VecFrameStack:
    # 8 stacked grayscale frames at 0.6s per step = 4.8s of temporal
    # context — enough to cover the full march of a ground unit from
    # bridge to tower (~3.5s) plus reaction time.
    return VecFrameStack(DummyVecEnv([ClashRoyaleEnv]), n_stack=8)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the newest SB3 auto-saved checkpoint, or None if missing.

    CheckpointCallback writes files named `{prefix}_{n}_steps.zip`; we
    pick the one with the largest step count (not mtime, so manual file
    shuffling doesn't confuse us).
    """
    candidates = list(checkpoint_dir.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"))
    if not candidates:
        return None

    def step_of(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-2])
        except (ValueError, IndexError):
            return -1

    best = max(candidates, key=step_of)
    return best if step_of(best) >= 0 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Clash Royale.")
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total global timesteps to train to (default: {DEFAULT_TIMESTEPS}).",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and start a new run from scratch.",
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a specific checkpoint zip to resume from. Overrides auto-detect.",
    )
    p.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the SB3 env sanity check at startup (avoids extra live clicks).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_check:
        sanity_env = ClashRoyaleEnv()
        check_env(sanity_env, warn=True, skip_render_check=True)

    env = build_env()

    resume_from: Path | None = None
    if not args.fresh:
        if args.resume_from is not None:
            if not args.resume_from.exists():
                raise FileNotFoundError(
                    f"--resume-from checkpoint not found: {args.resume_from}"
                )
            resume_from = args.resume_from
        else:
            resume_from = find_latest_checkpoint(CHECKPOINT_DIR)

    if resume_from is not None:
        print(f"[rl.train] Resuming from {resume_from}")
        model = PPO.load(
            str(resume_from),
            env=env,
            tensorboard_log=str(LOG_DIR),
        )
        already_done = int(model.num_timesteps)
        print(f"[rl.train] Prior timesteps: {already_done:,}")
    else:
        print("[rl.train] Starting fresh PPO run")
        model = PPO(
            # MultiInputPolicy handles Dict observation spaces: NatureCNN
            # for the "pixels"/"troops" image keys, separate Flatten
            # branches for "hand", "hand_affordable", "elixir", then
            # concatenates. Default CombinedExtractor — small enough to
            # learn quickly under tight step budgets.
            "MultiInputPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            # Small entropy bonus keeps exploration alive in the early
            # phase when the agent hasn't yet learned which card plays
            # are actually legal / useful. Default is 0.0.
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(LOG_DIR),
        )
        already_done = 0

    remaining = args.timesteps - already_done
    if remaining <= 0:
        print(
            f"[rl.train] Target of {args.timesteps:,} already reached "
            f"(model has {already_done:,} timesteps). Nothing to do."
        )
        model.save(str(CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_final"))
        return

    print(f"[rl.train] Training for {remaining:,} more timesteps "
          f"(target: {args.timesteps:,}).")

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix=CHECKPOINT_PREFIX,
    )

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
        tb_log_name=TB_LOG_NAME,
    )
    final_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_final"
    model.save(str(final_path))
    print(f"[rl.train] Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
