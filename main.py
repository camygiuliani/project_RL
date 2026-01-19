import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm,trange
import torch
import torch.nn.functional as F
import torch.nn as nn
import ale_py
import os
import time
from dqn_agent import DQN_Agent, ReplayBuffer
from wrappers import make_env
from ppo_agent import PPO_Agent
from sac_discrete import SACDiscreteAgent, SACDiscreteConfig




def main():
    parser = argparse.ArgumentParser(description='Run training of selected RL algorithm.')
    parser.add_argument('-dqn', '--dqn', action='store_true')
    parser.add_argument('-ddqn', '--ddqn', action='store_true')
    parser.add_argument('-ppo', '--ppo', action='store_true')
    parser.add_argument('-sac', '--sac', action='store_true')

    args = parser.parse_args()

    print("Starting training script...")
    env_id = "ALE/SpaceInvaders-v5"
    temp_env = make_env(env_id)
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


    ###
    ###     DQN Training
    ###
    if args.dqn:
        print("Starting DQN training...")
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
    
    ####
    ####        PPO training  
    ####  
    if args.ppo:
        print("Starting PPO training...")
        trainer = PPO_Agent(
            env_id="ALE/SpaceInvaders-v5",
            seed=0,
            rollout_len=128,
            n_epochs=4,
            batch_size=256,
            save_dir="runs/ppo_16_jan",
            eval_every=100000,
        )
        trainer.train(total_steps=100_000)
    
    ####
    ####        SAC (discrete) training
    ####
    if args.sac:
        print("Starting Discrete SAC training...")
        env = make_env(env_id)
        obs_shape = env.observation_space.shape  # (84,84,4)
        n_actions = env.action_space.n

        cfg = SACDiscreteConfig(
            replay_size=200_000,
            batch_size=128,
            alpha=0.2,
            start_steps=1_000,
            updates_per_step=1,
        )

        sac_agent = SACDiscreteAgent(obs_shape=obs_shape, n_actions=n_actions, device=device, cfg=cfg)

        total_steps = 10_000
        eval_every = 1_000
        log_every = 1_000
        save_dir = "runs/sac_discrete"
        os.makedirs(save_dir, exist_ok=True)

        obs, _ = env.reset(seed=0)
        ep_ret = 0.0
        next_eval = eval_every

        tqdm.write("Starting Discrete SAC training...")
        pbar = trange(1, total_steps + 1, desc="SAC training",leave=True)
        for step in  pbar:
            logs = None
            # act (stochastic)
            action = sac_agent.act(obs, deterministic=False)

            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(r)

            sac_agent.store(obs, action, r, next_obs, done)
            obs = next_obs

            if done:
                obs, _ = env.reset()
                ep_ret = 0.0

            # updates (off-policy)
            if sac_agent.total_steps > cfg.start_steps and sac_agent.can_update():
                logs=sac_agent.update_many(cfg.updates_per_step)
                    # stampa "umana" ogni tot step
            
            if step % log_every == 0 and logs is not None:
                tqdm.write(
                    f"[step {step}] q1={logs['q1_loss']:.2f} "
                    f"q2={logs['q2_loss']:.2f} actor={logs['actor_loss']:.2f}"
                )
                
            # checkpoint
            if step >= next_eval:
                ckpt = os.path.join(save_dir, f"sac_step_{step}.pt")
                sac_agent.save(ckpt)
                #print(f"[SAC] saved checkpoint: {ckpt}")
                tqdm.write(f"[SAC] saved checkpoint at step {step}")
                next_eval += eval_every

        env.close()
        final_path = os.path.join(save_dir, "sac_final.pt")
        sac_agent.save(final_path)
        print(f"[SAC] saved final checkpoint: {final_path}")
  
        
if __name__ == "__main__":
    main()
