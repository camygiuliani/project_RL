import os
import time
import numpy as np
from tqdm import tqdm
import torch

from wrappers import make_env
from replay_buffer import ReplayBuffer
from networks import DQNCNN
from dqn_agent import DQNAgent

def linear_eps(step, eps_start=1.0, eps_end=0.1, decay_steps=1_000_000):
    t = min(step / decay_steps, 1.0)
    return eps_start + t * (eps_end - eps_start)

def main():
    env_id = "ALE/SpaceInvaders-v5"
    seed = 0

    total_steps = 2_000_000
    learning_starts = 50_000
    train_freq = 4
    batch_size = 32
    buffer_size = 200_000
    target_update = 10_000

    gamma = 0.99
    lr = 1e-4
    double_dqn = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_id=env_id, seed=seed)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (84,84,4)

    q = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    tgt = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    agent = DQNAgent(q, tgt, n_actions, device, gamma=gamma, lr=lr, double_dqn=double_dqn)

    rb = ReplayBuffer(buffer_size, obs_shape)

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len = 0
    episode = 0

    os.makedirs("checkpoints", exist_ok=True)

    pbar = tqdm(range(1, total_steps + 1))
    for step in pbar:
        eps = linear_eps(step)
        action = agent.act(obs, eps)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rb.add(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if done:
            episode += 1
            obs, _ = env.reset()
            pbar.set_description(f"ep={episode} R={ep_ret:.1f} len={ep_len} eps={eps:.2f} rb={len(rb)}")
            ep_ret, ep_len = 0.0, 0

        # learning
        if step > learning_starts and step % train_freq == 0:
            batch = rb.sample(batch_size)
            loss = agent.update(batch)

        # target update
        if step % target_update == 0:
            agent.sync_target()

        # checkpoint
        if step % 200_000 == 0:
            ckpt_path = f"checkpoints/dqn_step_{step}.pt"
            agent.save(ckpt_path)

    env.close()

if __name__ == "__main__":
    main()
