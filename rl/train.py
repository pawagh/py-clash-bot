"""PPO training entry point.

Run from the repo root:

    uv run python -m rl.train

The env is NOT Gymnasium-pure — each step drives a real emulator, so
VecFrameStack + DummyVecEnv is the right default (SubprocVecEnv would
spawn multiple emulator controllers, which you should only do after
confirming your machine can host N Clash instances in parallel).
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from rl.env import ClashRoyaleEnv

CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./runs")


def build_env() -> VecFrameStack:
    return VecFrameStack(DummyVecEnv([ClashRoyaleEnv]), n_stack=4)


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check — this will actually talk to the emulator, so make sure
    # MEmu is running and Clash Royale is on the main menu before calling.
    sanity_env = ClashRoyaleEnv()
    check_env(sanity_env, warn=True, skip_render_check=True)

    env = build_env()

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="clash_ppo",
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    model.learn(total_timesteps=500_000, callback=checkpoint_cb)
    model.save(str(CHECKPOINT_DIR / "clash_ppo_final"))


if __name__ == "__main__":
    main()
