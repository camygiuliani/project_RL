import argparse
import csv
from datetime import datetime
from math import gamma
import os
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

    print("+++Importing configuration from config.yaml")
    cfg = load_config("config.yaml")
    parser = argparse.ArgumentParser(description='Run training of selected RL algorithm.')
   
    parser.add_argument('-ddqn', '--ddqn', action='store_true')
    parser.add_argument('-ppo', '--ppo', action='store_true')
    parser.add_argument('-sac', '--sac', action='store_true')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-eval', '--eval', action='store_true')
    parser.add_argument("--eval_episodes", type=int, default=cfg["eval"]["n_episodes"])
    parser.add_argument('-render', '--render', action='store_true')

    args = parser.parse_args()

    #######################################
    #####      ENVIRONMENT SETUP      ##### 
    #######################################
    env_id = cfg["env"]["id"]
    temp_env = make_env(env_id)
    n_actions = temp_env.action_space.n
    obs_shape = temp_env.observation_space.shape
    temp_env.close()
    
    print(f"Environment: {env_id} with detected action dimension {n_actions} and observation shape {obs_shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #######################################
    #####     AGENT INITIALIZATION    #####
    #######################################
        
    if args.ddqn:
            print("Initializing DDQN agent...")

            ddqn_agent = DDQN_Agent(env=env_id,n_channels= cfg['ddqn']['n_channels'],
                                   n_actions=n_actions, 
                                   device=device,
                                   gamma=cfg['ddqn']['gamma'], 
                                   lr=cfg['ddqn']['lr'],
                                   double_dqn=True)         
            info="DDQN"   
           
        
    if args.ppo:
            print("Initializing PPO agent...")
            ppo_agent = PPO_Agent(
                obs_shape=obs_shape,
                n_actions=n_actions,
                env_id=env_id,
                seed=cfg["ppo"]["seed"],
                rollout_len=cfg["ppo"]["rollout_steps"],
                n_epochs=cfg["ppo"]["epochs"],
                batch_size=cfg["ppo"]["batch_size"],
                save_dir=cfg["ppo"]["save_dir"],
                eval_every=cfg["ppo"]["eval_every"],
            )
            info="PPO"   
           
        
    if args.sac:
            print("Initializing SAC agent...")
            sac_agent = SACDiscrete_Agent(obs_shape=obs_shape,
                                            n_actions=n_actions,
                                            env_id=env_id,
                                            device=device,
                                            gamma= cfg['sac']['gamma'],
                                            tau = cfg['sac']['tau'],              # target soft update
                                            alpha = cfg['sac']['alpha'],              # entropy temperature (fixed, simple)
                                            actor_lr = cfg['sac']['actor_lr'],
                                            critic_lr= cfg['sac']['critic_lr'],
                                            batch_size = cfg['sac']['batch_size'],
                                            replay_size = cfg['sac']['replay_size'],
                                            start_steps = cfg['sac']['start_steps'],       # collect before updating heavily
                                            updates_per_step = cfg['sac']['updates_per_step'],       # how many gradient steps per env step after start
                                            max_grad_norm = cfg['sac']['max_grad_norm'])
            info="SAC"   

    ####################################
    #####      TRAINING BLOCK      #####
    ####################################
    start_train_time = time.time()
    if args.train:
        if args.ddqn:
              print("Starting DDQN training...\n")
              ddqn_agent.train(
                total_steps=cfg['ddqn']['total_steps'],
                l_start=cfg['ddqn']['l_start'],
                train_f=cfg['ddqn']['train_f'],
                batch_size=cfg['ddqn']['batch_size'],
                buffer_size= cfg['ddqn']['buffer_size'],
                target_update=cfg['ddqn']['target_update'],
                n_checkpoints=cfg['ddqn']['n_checkpoints'],
                log_every=cfg['ddqn']['log_every'],
                save_dir=cfg['ddqn']['save_dir'])


        if args.ppo:
              print("Starting PPO training...\n")
              ppo_agent.train(total_steps=cfg["ppo"]["total_steps"],
                              n_checkpoints=cfg["ppo"]["n_checkpoints"],
                              log_every=cfg["ppo"]["log_every"],
                              save_dir=cfg["ppo"]["save_dir"])
              
        if args.sac:
              print("Starting SAC training...\n")
              sac_agent.train(env=env_id,
                total_steps=cfg['sac']['total_steps'],
                log_every=cfg['sac']['log_every'],
                eval_every=cfg['sac']['eval_every'],
                save_dir=cfg['sac']['save_dir'])
              
        else:
            raise ValueError("Choose one algorithm for training: --ddqn or --ppo or --sac")
        
        end_train_time = time.time()
        train_duration = end_train_time - start_train_time
        print(f"Training with {info} completed in {train_duration/60:.2f} minutes.")
        
    ######################################
    #####      EVALUATION BLOCK      #####
    ######################################
    if args.eval:
        
        if args.ddqn:
            print("Starting DDQN evaluation...\n")
            ckpt=cfg["ddqn"]["path_best_model"]
            csv_dir=cfg["ddqn"]["save_dir"]

            agent=ddqn_agent
            
        
        elif args.ppo:
            print("Starting PPO evaluation...\n")
            ckpt=cfg["ppo"]["path_best_model"]
            csv_dir=cfg["ppo"]["save_dir"]
            agent=ppo_agent
           
           
        
        elif args.sac:
            print("Starting SAC evaluation...\n")
            ckpt=cfg["sac"]["path_best_model"]
            csv_dir=cfg["sac"]["save_dir"]

            agent=sac_agent
          
        else:
            raise ValueError("Choose one algorithm for evaluation: --ddqn or --ppo or --sac")

    
   
        # how much eposides to evaluate, if classes have this attribute
        if hasattr(agent, "eval_episodes"):
            agent.eval_episodes = args.eval_episodes

        # evaluation with loading the checkpoint
        n_ep = cfg["eval"]["n_episodes"] if not args.render else cfg["eval"]["n_episodes_render"]
        out = agent.eval(seed=cfg["eval"]["seed"],
                        n_episodes=n_ep,
                        path=ckpt, render_mode=None if not args.render else "human")

        
        mean_r, std_r = out

        print(f"[EVAL] algo={('ddqn' if args.ddqn else 'ppo' if args.ppo else 'sac')} ckpt={ckpt} "
            f"episodes={n_ep} mean={mean_r:.2f} std={std_r:.2f}")


        #csv logging in a file

        date_csv = datetime.now().strftime("%Y_%m_%d")
        time_csv = datetime.now().strftime("%H_%M_%S")
        outdir_csv = os.path.join(csv_dir, date_csv)
        os.makedirs(outdir_csv, exist_ok=True)


        csv_path =  os.path.join(outdir_csv, f"metrics_eval_{time_csv}.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["algo", "episodes", "mean_return", "std_return"])
            writer.writerow(["ddqn", n_ep, mean_r, std_r])

  
    return 0
        
if __name__ == "__main__":
    main()
