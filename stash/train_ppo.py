# train_ppo.py
import os
from typing import Callable

from wrappers import make_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


def _make_env_fn(env_id: str, seed: int, rank: int = 0) -> Callable[[], object]:
    def _init():
        env = make_env(env_id=env_id, seed=seed + rank, frame_skip=4)
        env = Monitor(env)
        return env
    return _init


def train_ppo(
    env_id: str = "ALE/SpaceInvaders-v5",
    seed: int = 0,
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    log_dir: str = "runs/ppo",
    eval_freq: int = 100_000,
    eval_episodes: int = 20,
    save_name: str = "ppo_spaceinvaders",
    tensorboard: bool = False,
):
    os.makedirs(log_dir, exist_ok=True)

    # Vec env (SB3 wants vectorized env)
    train_env = DummyVecEnv([_make_env_fn(env_id, seed, i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecTransposeImage(train_env)  # HWC->CHW (84,84,4) -> (4,84,84)

    eval_env = DummyVecEnv([_make_env_fn(env_id, seed + 10_000, 0)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=2.5e-4,
        n_steps=128,        # per env
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir if tensorboard else None,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(eval_freq // 2, 10_000),
        save_path=log_dir,
        name_prefix=f"{save_name}_ckpt",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, eval_cb], progress_bar=True)

    final_path = os.path.join(log_dir, f"{save_name}.zip")
    model.save(final_path)
    print(f"\nSaved final PPO model to: {final_path}")

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    print(f"Final eval over 50 episodes: mean={mean_r:.2f}  std={std_r:.2f}")

    train_env.close()
    eval_env.close()
    return final_path
