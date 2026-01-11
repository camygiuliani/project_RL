import numpy as np
import torch
from wrappers import make_env
from dqn_agent import DQN_Agent, DQNCNN
import ale_py 

def main():
    env_id = "ALE/SpaceInvaders-v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = make_env(env_id=env_id, seed=123)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    q = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    tgt = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    agent = DQN_Agent(q, tgt, n_actions, device, double_dqn=True)

    agent.load("checkpoints/dqn_step_200000.pt")  # cambia path

    n_episodes = 10
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

    print("Mean return:", float(np.mean(returns)))
    env.close()

if __name__ == "__main__":
    main()
