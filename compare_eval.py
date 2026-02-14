import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from ddqn import DDQN_Agent
from ppo import PPO_Agent
from sac import SACDiscrete_Agent
from utils import load_config
from wrappers import make_env

def compare_algorithms(env_id: str,ddqn_path: str, ppo_path: str, sac_path: str,
                       n_episodes: int = 10, cfg = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device)
    print(f"--- Starting Comparison on {device} using {n_episodes} episodes ---")

    temp_env = make_env(env_id)
    obs_shape = temp_env.observation_space.shape
    n_actions = temp_env.action_space.n
    temp_env.close()
    
    
    print(f"Environment: {env_id} | Obs: {obs_shape} | Actions: {n_actions}")

    # Store results here
    results = {
        "DDQN": {"mean": 0.0, "std": 0.0, "color": "#1f77b4"}, 
        "PPO":  {"mean": 0.0, "std": 0.0, "color": "#ff7f0e"}, 
        "SAC":  {"mean": 0.0, "std": 0.0, "color": "#2ca02c"}  
    }

    if ddqn_path and os.path.exists(ddqn_path):
        print(f"\nEvaluating DDQN (ckpt: {os.path.basename(ddqn_path)})...")
        agent = DDQN_Agent(env=env_id,
                            n_channels= cfg['ddqn']['n_channels'],
                            obs_shape=obs_shape,
                            n_actions=n_actions, 
                            device=device,
                            gamma=cfg['ddqn']['gamma'], 
                            lr=cfg['ddqn']['lr'],
                            double_dqn=True)        
        
        mean_r, std_r = agent.eval(seed=cfg["eval"]["seed"], n_episodes=n_episodes, path=ddqn_path)
        results["DDQN"]["mean"] = mean_r
        results["DDQN"]["std"] = std_r
    else:
        print(f"Skipping DDQN: Path not found or None ({ddqn_path})")

    if ppo_path and os.path.exists(ppo_path):
        print(f"\nEvaluating PPO (ckpt: {os.path.basename(ppo_path)})...")
        agent = PPO_Agent(obs_shape=obs_shape,
                        n_envs=cfg["ppo"]["n_envs"],
                        n_actions=n_actions, 
                        env_id=env_id,
                        device=device,
                        lr= cfg["ppo"]["lr"],
                        gamma=cfg["ppo"]["gamma"],
                        gae_lambda=cfg["ppo"]["gae_lambda"],
                        rollout_len=cfg["ppo"]["rollout_steps"])
        
        mean_r, std_r = agent.eval(seed=cfg["eval"]["seed"], n_episodes=n_episodes, path=ppo_path)
        results["PPO"]["mean"] = mean_r
        results["PPO"]["std"] = std_r
    else:
        print(f"Skipping PPO: Path not found or None ({ppo_path})")

    if sac_path and os.path.exists(sac_path):
        print(f"\nEvaluating SAC (ckpt: {os.path.basename(sac_path)})...")
        agent = SACDiscrete_Agent(obs_shape=obs_shape,
                                n_actions=n_actions,
                                device=device,
                                env_id=env_id,
                                actor_lr = cfg['sac']['actor_lr'],
                                critic_lr= cfg['sac']['critic_lr'],
                                replay_size = cfg['sac']['replay_size'])
        
        mean_r, std_r = agent.eval(seed=cfg["eval"]["seed"], n_episodes=n_episodes, path=sac_path)
        results["SAC"]["mean"] = mean_r
        results["SAC"]["std"] = std_r
    else:
        print(f"Skipping SAC: Path not found or None ({sac_path})")

    print("\nGenerating comparison graph...")
    
    names = []
    means = []
    stds = []
    colors = []

    for algo, data in results.items():
        if data["mean"] != 0 or data["std"] != 0:
            names.append(algo)
            means.append(data["mean"])
            stds.append(data["std"])
            colors.append(data["color"])

    if not names:
        print("No results to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(names, means, yerr=stds, capsize=10, color=colors, alpha=0.9, width=0.6)

    plt.title(f'Performance Comparison on {env_id}\n({n_episodes} Evaluation Episodes)', fontsize=14)
    plt.ylabel('Average Return', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + (max(stds) * 0.1), 
            f'{height:.2f}', 
            ha='center', va='bottom', fontweight='bold'
        )

    plt.tight_layout()
    
    # Save the plot
    path = os.path.join("./outputs")
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, "comparison_results.png")
    plt.savefig(output_file)
    print(f"Graph saved as '{output_file}'")

if __name__ == "__main__":
    
    cfg = load_config("config.yaml")
    compare_algorithms(
        env_id="ALE/SpaceInvaders-v5",
        ddqn_path=cfg["ddqn"]["path_best_model"],
        ppo_path=cfg["ppo"]["path_best_model"],
        sac_path=cfg["sac"]["path_best_model"],
        cfg = cfg,
        n_episodes=50
    )