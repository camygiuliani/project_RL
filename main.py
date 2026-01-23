import argparse
import numpy as np
from tqdm import tqdm,trange
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import load_config
from ddqn import DDQN_Agent
from wrappers import make_env
from ppo import PPO_Agent
from sac import SACDiscrete_Agent, SACDiscreteConfig    




def main():
    parser = argparse.ArgumentParser(description='Run training of selected RL algorithm.')
   
    parser.add_argument('-ddqn', '--ddqn', action='store_true')
    parser.add_argument('-ppo', '--ppo', action='store_true')
    parser.add_argument('-sac', '--sac', action='store_true')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-eval', '--eval', action='store_true')
    parser.add_argument('-render', '--render', action='store_true')

    args = parser.parse_args()

    print("Starting training script...importing configuration from config.yaml")
    cfg = load_config("config.yaml")
    env_id = cfg["env"]["id"]

    temp_env = make_env(env_id)
    n_actions = temp_env.action_space.n
    obs_shape = temp_env.observation_space.shape
    temp_env.close()
    
    print(f"Environment: {env_id}")
    print(f"Detected Action Dimension: {n_actions}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    

    #TRAINING SECTION 
    if args.train:
        
        if args.ddqn:
            print("Starting DQN training...")

            ddqn_agent = DDQN_Agent(env=env_id,n_channels= cfg['ddqn']['n_channels'],
                                   n_actions=n_actions, 
                                   device=device,
                                   gamma=cfg['ddqn']['gamma'], 
                                   lr=cfg['ddqn']['lr'],
                                   double_dqn=True)            
            ddqn_agent.train(
                total_steps=cfg['ddqn']['total_steps'],
                l_start=cfg['ddqn']['l_start'],
                train_f=cfg['ddqn']['train_f'],
                batch_size=cfg['ddqn']['batch_size'],
                buffer_size= cfg['ddqn']['buffer_size'],
                target_update=cfg['ddqn']['target_update'],
                n_checkpoints=cfg['ddqn']['n_checkpoints'],
                save_dir=cfg['ddqn']['save_dir'])
        
        if args.ppo:

            #PPO hyperparameters
            
            print("Starting PPO training...")
            ppo_agent = PPO_Agent(
                obs_shape=obs_shape,
                n_actions=n_actions,
                env_id=env_id,
                seed=cfg,
                rollout_len=cfg["ppo"]["rollout_steps"],
                n_epochs=cfg["ppo"]["epochs"],
                batch_size=cfg["ppo"]["batch_size"],
                save_dir=cfg["ppo"]["save_dir"],
                eval_every=cfg["ppo"]["eval_every"],
            )
            ppo_agent.train(total_steps=cfg["ppo"]["total_steps"])
        
        

        if args.sac:
            print("Starting Discrete SAC training...")
            sac_agent = SACDiscrete_Agent(obs_shape=obs_shape,
                                          n_actions=n_actions,
                                          env_id=env_id,
                                          device=device)
            

    #EVALUATION SECTION
    if args.eval:
        
        if args.ddqn:
            #evaluate_dqn(model_hyperparams=dqn_hyperparams, model_path="runs/dqn/dqn_1000000.pt",
            #             env_id=env_id, n_episodes=10, device=device)  
        
           # TODO: implement evaluation function for DDQN
        
            return 0  
        
        
        if args.ppo:
            # TODO: implement evaluation function for PPO
            return 0  
        
        if args.sac:
            # TODO: implement evaluation function for SAC
            return 0  
  
        
if __name__ == "__main__":
    main()
