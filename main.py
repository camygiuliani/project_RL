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
import yaml
from dqn import DQN_Agent, ReplayBuffer
from wrappers import make_env
from ppo import PPO_Agent
from sac import SACDiscreteAgent, SACDiscreteConfig


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate_dqn(model_path, model_hyperparams, env_id, n_episodes, device):
    env = make_env(env_id=env_id, seed=123)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    agent = DQN_Agent(**model_hyperparams)
    agent.load(model_path)

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        R = 0.0
        while not done:
            a = agent.act(obs, eps=0.0)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            R += r
        returns.append(R)
        print(f"Episode {ep+1}: return={R:.1f}")

    mean_return = float(np.mean(returns))
    print("Mean return:", mean_return)
    env.close()
    return mean_return


def main():
    parser = argparse.ArgumentParser(description='Run training of selected RL algorithm.')
    parser.add_argument('-dqn', '--dqn', action='store_true')
    parser.add_argument('-ddqn', '--ddqn', action='store_true')
    parser.add_argument('-ppo', '--ppo', action='store_true')
    parser.add_argument('-sac', '--sac', action='store_true')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-eval', '--eval', action='store_true')
    parser.add_argument('-render', '--render', action='store_true')

    args = parser.parse_args()

    print("Starting training script...")
    cfg = load_config("config.yaml")
    env_id = cfg["env"]["id"]

    temp_env = make_env(env_id)
    n_actions = temp_env.action_space.n
    temp_env.close()
    
    print(f"Environment: {env_id}")
    print(f"Detected Action Dimension: {n_actions}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    

    #TRAINING SECTION 
    if args.train:
        
        if args.dqn:
            print("Starting DQN training...")

            dqn_hyperparams = {
                "n_channels": 4,              
                "n_actions": n_actions,       
                "device": device,             
                "env": env_id,                
                "gamma": 0.99,                
                "lr": 1e-4,                   
                "double_dqn": True                         
            }

            dqn_training_params = {
                "total_steps": 1_000_000,
                "l_start": 50_000,
                "train_f": 4,
                "batch_size": 32,
                "buffer_size": 200_000,
                "target_update": 10_000,
                "n_checkpoints": 5,
                "save_dir": "runs/dqn"
            }

            dqn_agent = DQN_Agent(**dqn_hyperparams)                        
            dqn_agent.train(**dqn_training_params)
        
        if args.ppo:

            #PPO hyperparameters
            
            print("Starting PPO training...")
            ppo_agent = PPO_Agent(
                env_id=env,
                seed=cfg,
                rollout_len=cfg["ppo"]["rollout_steps"],
                n_epochs=cfg["ppo"]["epochs"],
                batch_size=cfg["ppo"]["batch_size"],
                save_dir=cfg["ppo"]["save_dir"],
                eval_every=cfg["ppo"]["eval_every"],
            )
            ppo_agent.train(total_steps=cfg["ppo"]["total_steps"])
        
        #Si deve sistemare mettendo il training nella classe come tutti gli altri

        if args.sac:
            print("Starting Discrete SAC training...")
            env = make_env(env_id)
            obs_shape = env.observation_space.shape  # (84,84,4)

            cfg_s = SACDiscreteConfig(
                replay_size=cfg["sac"]["replay_size"],
                batch_size= cfg["sac"]["batch_size"],
                alpha=cfg["sac"]["alpha"],
                start_steps=cfg["sac"]["start_steps"],
                updates_per_step=cfg["sac"]["updates_per_step"],
            )

            sac_agent = SACDiscreteAgent(obs_shape=obs_shape, n_actions=n_actions, device=device, cfg=cfg_s)

            total_steps = cfg["sac"]["total_steps"]
            eval_every = cfg["sac"]["eval_every"]
            log_every = cfg["sac"]["log_every"]
            save_dir = cfg["sac"]["save_dir"]
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

    #EVALUATION SECTION
    if args.eval:
        
        if args.dqn:
            evaluate_dqn(model_hyperparams=dqn_hyperparams, model_path="runs/dqn/dqn_1000000.pt",
                         env_id=env_id, n_episodes=10, device=device)  
        
        if args.ddqn:
            return 0  
        
        if args.ppo:
            return 0  
        
        if args.sac:
            return 0  
  
        
if __name__ == "__main__":
    main()
