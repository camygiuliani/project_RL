import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import ale_py
from replay_buffer import ReplayBuffer
import os
import time
from dqn_agent import DQN_Agent
from wrappers import make_env
from train_ppo import train_ppo



def main():
    env_id = "ALE/SpaceInvaders-v5"
    temp_env = gym.make(env_id)
    action_dim = temp_env.action_space.n
    temp_env.close()
    
    print(f"Environment: {env_id}")
    print(f"Detected Action Dimension: {action_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #DQN init
    dqn_channels = 4 
    dqn_gamma=0.99
    dqn_lr=1e-4
    double_dqn=True

    #DQN training
    dqn_seed = 0
    dqn_steps = 2_000_000
    dqn_start = 50_000
    dqn_f = 4
    dqn_batch_size = 32
    dqn_buffer_size = 200_000
    dqn_target_update = 10_000
    dqn_n_checkpoints = 5 

    dqn_agent = DQN_Agent(n_channels=dqn_channels, n_actions=action_dim, 
                           device=device, env=env_id, gamma=dqn_gamma, lr =dqn_lr,
                           double_dqn=double_dqn)
    

    dqn_agent.train(batch_size=dqn_batch_size, buffer_size=dqn_buffer_size, total_steps=dqn_steps, l_start=dqn_start,
                    train_f=dqn_f, target_update=dqn_target_update, n_checkpoints=dqn_n_checkpoints)
    
    #PPO training
    run_ppo = True 
    if run_ppo:
        train_ppo(
            env_id="ALE/SpaceInvaders-v5",
            seed=0,
            total_timesteps=2_000_000,
            n_envs=8,
            log_dir="runs/ppo",
            tensorboard=True,
        )
    
    return
