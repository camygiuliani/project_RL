# sac_discrete.py
from datetime import datetime
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils import load_config
from wrappers import make_env


class SacCNN(nn.Module):
    """Atari-style CNN encoder (like DQN) + MLP head."""
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # For 84x84 Atari convs -> 7x7x64 = 3136
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z)
        return z
    
class CNNEncoder(nn.Module):
    """
    Atari-style CNN encoder (like DQN).
    Input: (N,4,84,84) float in [0,1]
    Output: (N,512)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # 84x84 -> 7x7x64 = 3136
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )

    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z)
        return z


class DiscreteActor(nn.Module):
    """
    Actor: outputs logits -> Categorical policy over discrete actions.
    """
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.enc = CNNEncoder(in_channels)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        z = self.enc(x)
        logits = self.head(z)
        return logits


class DiscreteCritic(nn.Module):
    """
    Critic: outputs Q(s, a) for all actions -> (N, n_actions)
    """
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.enc = CNNEncoder(in_channels)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        z = self.enc(x)
        q = self.head(z)
        return q



@dataclass
class SACDiscreteConfig:
    gamma: float = 0.99
    tau: float = 0.005              # target soft update
    alpha: float = 0.2              # entropy temperature (fixed, simple)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 128
    replay_size: int = 200_000
    start_steps: int = 20_000       # collect before updating heavily
    updates_per_step: int = 1       # how many gradient steps per env step after start
    max_grad_norm: float = 10.0     # optional safety clip


class SACDiscrete_Agent:
    """
    Discrete SAC (Atari-friendly).
    - Off-policy (ReplayBuffer)
    - Actor: pi(a|s) categorical
    - Critic: two Q networks, target Q networks
    """

    def __init__(self, obs_shape, n_actions: int, device: torch.device, 
                    env_id="ALE/SpaceInvaders-v5",
                    gamma: float = 0.99,
                    tau: float = 0.005,              # target soft update
                    alpha: float = 0.2,              # entropy temperature (fixed, simple)
                    actor_lr: float = 3e-4,
                    critic_lr: float = 3e-4,
                    batch_size: int = 128,
                    replay_size: int = 200_000,
                    start_steps: int = 20_000,       # collect before updating heavily
                    updates_per_step: int = 1,       # how many gradient steps per env step after start
                    max_grad_norm: float = 10.0):
        
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        
        
        # loading yaml config
        self.cfg = load_config("config.yaml")
        self.env_id = env_id

        C = obs_shape[2]
        assert C == 4, "Expected stacked frames (84,84,4)"

        # Networks
        self.actor = SacCNN(C, n_actions).to(device)
        self.q1 = SacCNN(C, n_actions).to(device)
        self.q2 = SacCNN(C, n_actions).to(device)

        self.q1_tgt = SacCNN(C, n_actions).to(device)
        self.q2_tgt = SacCNN(C, n_actions).to(device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        self.q1_tgt.eval()
        self.q2_tgt.eval()

        # Optims
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg["sac"]["actor_lr"])
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.cfg["sac"]["critic_lr"])
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.cfg["sac"]["critic_lr"])

        # Buffer
        self.replay = ReplayBuffer(obs_shape, self.cfg["sac"]["replay_size"], device)

        # Counters
        self.total_steps = 0
        self.total_updates = 0

    @torch.no_grad()
    def act(self, obs_uint8: np.ndarray, deterministic: bool = False) -> int:
        """
        obs_uint8: (84,84,4) uint8
        """
        x = torch.from_numpy(obs_uint8).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        logits = self.actor(x)
        if deterministic:
            a = int(torch.argmax(logits, dim=1).item())
            return a
        dist = torch.distributions.Categorical(logits=logits)
        a = int(dist.sample().item())
        return a
    
    def _soft_update(self, net: nn.Module, tgt: nn.Module):
        tau = self.cfg.tau
        for p, pt in zip(net.parameters(), tgt.parameters()):
            pt.data.mul_(1.0 - tau)
            pt.data.add_(tau * p.data)

    def _policy_probs_and_logp(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: (N,A)
        returns:
          probs: (N,A)
          log_probs: (N,A)
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def can_update(self) -> bool:
        return len(self.replay) >= self.cfg.batch_size

    def update(self) -> dict:
        """
        One SAC update step (critic(s) + actor + target soft update).
        Returns a dict of losses for logging.
        """
        if not self.can_update():
            return {}

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.cfg.batch_size)
        gamma = self.cfg.gamma
        alpha = self.cfg.alpha

        
        #1) Soft policy evaluation step
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs, next_logp = self._policy_probs_and_logp(next_logits)

            q1_next = self.q1_tgt(next_obs)                 
            q2_next = self.q2_tgt(next_obs)

            #Prevent overestimation of Q-values.
            #using two Q-functions to mitigate positive bias
            qmin_next = torch.min(q1_next, q2_next)       

            #1.1) Compute Value of next state V(s') explicitly
            #     V(s') = sum [pi(a'|s')*( Q_min(s', a') - alpha * log pi(a'|s'))]

            v_next = torch.sum(next_probs * (qmin_next - alpha * next_logp), dim=1) 
            #1.2) Compute Soft Bellman Target 
            #     y = r + gamma*(1-done)*V(s')
            y = rewards + gamma * (1.0 - dones) * v_next      

        # -----------------------
        # Critic losses
        # We regress Q(s,a) to y.
        # -----------------------
        q1_all = self.q1(obs)                              
        q2_all = self.q2(obs)
        q1_sa = q1_all.gather(1, actions.view(-1, 1)).squeeze(1)
        q2_sa = q2_all.gather(1, actions.view(-1, 1)).squeeze(1)

        #1.3) minimize MSE loss 
        q1_loss = F.mse_loss(q1_sa, y)
        q2_loss = F.mse_loss(q2_sa, y)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.max_grad_norm)
        self.q2_opt.step()

        logits = self.actor(obs)
        probs, logp = self._policy_probs_and_logp(logits)

        with torch.no_grad():
            q1_pi = self.q1(obs)
            q2_pi = self.q2(obs)
            qmin_pi = torch.min(q1_pi, q2_pi)
        ####################################

        #2) Actor update     
        #   Calculate gradients for Actor
        #   J_pi = sum [ pi * (alpha * log_pi - Q) ]
        actor_loss = torch.sum(probs * (alpha * logp - qmin_pi), dim=1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        #3) Target soft updates
        #   Stabilize training by slowly updating target networks.
        self._soft_update(self.q1, self.q1_tgt)
        self._soft_update(self.q2, self.q2_tgt)

        self.total_updates += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha),
        }

    def update_many(self, n_updates: int) -> dict:
        """
        Run multiple update steps (useful: updates_per_step).
        """
        logs = {}
        for _ in range(n_updates):
            out = self.update()
            if not out:
                break
            logs = out
        return logs
    
    def train(self, env, total_steps: int, eval_every: int, log_every: int, save_dir: str):
        env = make_env(self.env_id)

        # creating directory with date for saving runs
        date = datetime.now().strftime("%Y_%m_%d")
        outdir_runs = os.path.join(save_dir, date)
        os.makedirs(outdir_runs, exist_ok=True)
        # final path for saving the final model
        final_path = os.path.join(outdir_runs, f"sac_{total_steps}.pt")

        obs, _ = env.reset(seed=0)
        ep_ret = 0.0
        next_eval = eval_every

        os.makedirs(f"checkpoints/sac", exist_ok=True)

        tqdm.write("Starting Discrete SAC training...")
        pbar = trange(1, total_steps + 1, desc="SAC training",leave=True)
        for step in  pbar:
            logs = None
            # act (stochastic)
            action = self.act(obs, deterministic=False)

            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(r)

            self.replay.add(obs, action, r, next_obs, done)
            self.total_steps += 1
            obs = next_obs

            if done:
                obs, _ = env.reset()
                ep_ret = 0.0

            # updates (off-policy)
            if self.total_steps > self.cfg["sac"]["start_steps"] and self.can_update():
                logs=self.update_many(self.cfg["sac"]["updates_per_step"])
                    # stampa "umana" ogni tot step
            
            if step % log_every == 0 and logs is not None:
                tqdm.write(
                    f"[step {step}] q1={logs['q1_loss']:.2f} "
                    f"q2={logs['q2_loss']:.2f} actor={logs['actor_loss']:.2f}"
                )
                
            # checkpoint
            if step >= next_eval:
                ckpt_dir= self.cfg["sac"]["checkpoints_dir"]

                date = datetime.now().strftime("%Y_%m_%d")
                time = datetime.now().strftime("%H_%M_%S")
                outdir_ckpt = os.path.join(ckpt_dir, date)
                os.makedirs(outdir_ckpt, exist_ok=True)
                ckpt_path = os.path.join(outdir_ckpt, f"sac_step_{step}_{time}.pt")

                self.save(ckpt_path)

                tqdm.write(f"[SAC] saved checkpoint at step {step}")
                next_eval += eval_every

        env.close()
        self.save(final_path)
        print(f"[SAC] saved final checkpoint: {final_path}")

    def eval(self, seed, n_episodes: int = 10, path: str = None):    
        if path is not None:
            self.load(path)
        else:
            print("No checkpoint provided.")
            return -1

        env = make_env(env_id=self.env_id, seed=seed)
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            R = 0.0
            while not done:
                a = self.act(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                R += r
            returns.append(R)
            print(f"Episode {ep+1}: return={R:.1f}")

        mean_return = round(float(np.mean(returns)), 3)
        std_return  = round(float(np.std(returns)), 3)
        print(f"Mean return {mean_return} and std {std_return} over {n_episodes} episodes")
        env.close()
        return mean_return, std_return

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_tgt": self.q1_tgt.state_dict(),
                "q2_tgt": self.q2_tgt.state_dict(),
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_tgt.load_state_dict(ckpt["q1_tgt"])
        self.q2_tgt.load_state_dict(ckpt["q2_tgt"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.total_updates = int(ckpt.get("total_updates", 0))
        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_tgt.to(self.device)
        self.q2_tgt.to(self.device)
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_tgt.eval()
        self.q2_tgt.eval()


class ReplayBuffer:
    def __init__(self, obs_shape, size: int, device: torch.device):
        """
        obs_shape: (H,W,C) with C=4 (uint8)
        """
        self.size = int(size)
        self.device = device

        H, W, C = obs_shape
        self.obs = np.zeros((self.size, H, W, C), dtype=np.uint8)
        self.next_obs = np.zeros((self.size, H, W, C), dtype=np.uint8)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        n = len(self)
        assert n > 0, "ReplayBuffer is empty"
        idx = np.random.randint(0, n, size=batch_size)

        obs = torch.from_numpy(self.obs[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        next_obs = torch.from_numpy(self.next_obs[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)

        return obs, actions, rewards, next_obs, dones
