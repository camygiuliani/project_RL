from datetime import datetime
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils import load_config,save_training_csv
from wrappers import make_env, make_env_eval


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
    
    
class SACDiscrete_Agent:
    """
    Discrete SAC (Atari-friendly).
    - Off-policy (ReplayBuffer)
    - Actor: pi(a|s) categorical
    - Critic: two Q networks, target Q networks
    """

    def __init__(self, obs_shape, n_actions: int, device: torch.device, 
                    env_id="ALE/SpaceInvaders-v5",
                    actor_lr: float = 3e-4,
                    critic_lr: float = 3e-4,
                    alpha_lr: float = 3e-4,
                    replay_size: int = 100000):
        
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.env_id = env_id

        if obs_shape[0] == 4:
            C = obs_shape[0]
        # Se shape è (84, 84, 4) -> C=4 a indice 2
        elif obs_shape[-1] == 4:
            C = obs_shape[-1]
        else:
            raise ValueError(f"Formato osservazione non supportato: {obs_shape}")

        self.target_entropy = 0.3 * np.log(n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        
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
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr= actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr= critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr= critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Buffer
        self.replay = ReplayBuffer(obs_shape, replay_size, device)

        # Counters
        self.total_steps = 0
        self.total_updates = 0

       



    def preprocess_obs(self, obs_uint8):
        x = torch.as_tensor(obs_uint8, device=self.device).float()

        # Single obs: CHW (4,84,84) or HWC (84,84,4)
        if x.ndim == 3:
            if x.shape[0] in (1, 4):                 # CHW
                x = x.unsqueeze(0)                   # (1,C,H,W)
            elif x.shape[-1] in (1, 4):              # HWC
                x = x.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
            else:
                raise ValueError(f"Unknown obs shape: {tuple(x.shape)}")
        return x / 255.0
    
    @torch.no_grad()
    def act(self, obs_uint8: np.ndarray, deterministic: bool = False) -> int:
        """
        obs_uint8: (84,84,4) uint8
        """
        x = self.preprocess_obs(obs_uint8)
        
        logits = self.actor(x)
        if deterministic:
            a = int(torch.argmax(logits, dim=1).item())
            return a
        dist = torch.distributions.Categorical(logits=logits)
        a = int(dist.sample().item())
        return a
    
    def _soft_update(self, net: nn.Module, tgt: nn.Module, tau: float):
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

    def can_update(self, batch_size: int) -> bool:
        return len(self.replay) >= batch_size

    def update(self, max_grad_norm: float, batch_size: int, gamma, tau) -> dict:
        """
        One SAC update step (critic(s) + actor + target soft update).
        Returns a dict of losses for logging.
        """
        if not self.can_update(batch_size= batch_size):
            return {}

        obs, actions, rewards, next_obs, dones = self.replay.sample(batch_size)
        
        alpha = self.log_alpha.exp().item()
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
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_grad_norm)
        self.q2_opt.step()

        logits = self.actor(obs)
        probs, logp = self._policy_probs_and_logp(logits)

        with torch.no_grad():
            q1_pi = self.q1(obs)
            q2_pi = self.q2(obs)
            qmin_pi = torch.min(q1_pi, q2_pi)
            entropy = -torch.sum(probs * logp, dim=1)
        ####################################

        #2) Actor update     
        #   Calculate gradients for Actor
        #   J_pi = sum [ pi * (alpha * log_pi - Q) ]
        actor_loss = torch.sum(probs * (alpha * logp - qmin_pi), dim=1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_opt.step()

        #clamping log_alpha for stability and to prevent extreme values
        with torch.no_grad():
            self.log_alpha.clamp_(min=-5.0, max=2.0)
   
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        #3) Target soft updates
        #   Stabilize training by slowly updating target networks.
        self._soft_update(net= self.q1, tgt= self.q1_tgt, tau= tau)
        self._soft_update(net= self.q2, tgt= self.q2_tgt, tau= tau)

        self.total_updates += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()), # Log this
            "alpha": alpha, # Log the current alpha value
            "entropy": float(entropy.mean().item()) # Useful to track
        }

    def update_many(self, n_updates: int, gamma: float, tau: float, 
                    max_grad_norm: float, batch_size: int) -> dict:
        """
        Run multiple update steps (useful: updates_per_step).
        """
        logs = {}
        for _ in range(n_updates):
            out = self.update(max_grad_norm= max_grad_norm, batch_size= batch_size, gamma= gamma, tau= tau)
            if not out:
                break
            logs = out
        return logs
    
    def train(self, env, total_steps: int, start_steps: int, log_every: int, 
              updates_per_step: int, checkpoint_dir: str, 
              n_checkpoints: int, save_dir: str, max_grad_norm: float, batch_size: int,
              gamma: float = 0.99, tau: float = 0.005):
        
        seed = 0
        threshold = total_steps // n_checkpoints if n_checkpoints > 0 else 0
        c_threshold = threshold

        date = datetime.now().strftime("%Y_%m_%d")
        outdir_runs = os.path.join(save_dir, date)
        os.makedirs(outdir_runs, exist_ok=True)
        final_path = os.path.join(outdir_runs, f"sac_{total_steps}.pt")
        final_path_csv = os.path.join(outdir_runs, f"metrics_train_{total_steps}.csv")

        print(f"Using device: {self.device}")
        env = make_env(self.env_id, seed=seed, scale_reward=False)  # Scale rewards for SAC
        obs, _ = env.reset(seed=seed)

        os.makedirs(f"checkpoints/sac", exist_ok=True)

        history = []
        returns_window = []
        episode = 0
        ep_ret = 0.0
        ep_len=0
        ep_q1_losses=[]
        ep_q2_losses=[]
        ep_actor_losses=[]
        ep_alphas=[]
        ep_entropies=[]

        tqdm.write("Starting Discrete SAC training...")
        pbar = trange(1, total_steps + 1, desc="SAC training",leave=True)
        for step in  pbar:
            logs = None
            # act (stochastic)
            action = self.act(obs, deterministic=False)

            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(r)
            ep_len += 1

            self.replay.add(obs, action, r, done)
            self.total_steps += 1
            obs = next_obs

            if done:
                episode += 1
                returns_window.append(ep_ret)
                avg100 = sum(returns_window[-100:]) / min(len(returns_window), 100)

                # mean of losses per episode
                q1_mean = sum(ep_q1_losses)/len(ep_q1_losses) if ep_q1_losses else None
                q2_mean = sum(ep_q2_losses)/len(ep_q2_losses) if ep_q2_losses else None
                actor_mean = sum(ep_actor_losses)/len(ep_actor_losses) if ep_actor_losses else None
                alpha_mean = sum(ep_alphas)/len(ep_alphas) if ep_alphas else None
                entropy_mean = sum(ep_entropies)/len(ep_entropies) if ep_entropies else None

                history.append({
                    "env_step": int(step),
                    "episode": int(episode),
                    "ep_len": int(ep_len),
                    "episodic_return": float(ep_ret),
                    "avg_return_100": float(avg100),
                    "q1_loss_ep_mean": q1_mean,
                    "q2_loss_ep_mean": q2_mean,
                    "actor_loss_ep_mean": actor_mean,
                    "alpha_ep_mean": alpha_mean,
                    "entropy_ep_mean": entropy_mean,
                    "updates_in_episode": int(len(ep_q1_losses)),
                })
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0
                ep_q1_losses.clear()
                ep_q2_losses.clear()
                ep_actor_losses.clear()
                ep_alphas.clear()

            # updates (off-policy)
            if self.total_steps > start_steps and self.can_update(batch_size= batch_size):
                logs=self.update_many(n_updates=updates_per_step, gamma=gamma, tau= tau,
                                      max_grad_norm=max_grad_norm, batch_size=batch_size)
                if logs is not None:
                    ep_q1_losses.append(float(logs["q1_loss"]))
                    ep_q2_losses.append(float(logs["q2_loss"]))
                    ep_actor_losses.append(float(logs["actor_loss"]))
                    ep_alphas.append(float(logs["alpha"]))
                    ep_entropies.append(float(logs["entropy"]))


            
            if step % log_every == 0:
                if logs is not None:
                   tqdm.write(      
                       f"[step {step}] "
                       f"Avg100={avg100:.3f}, "
                       f"q1={logs['q1_loss']:.3f} "
                       f"q2={logs['q2_loss']:.3f} "
                       f"Actor={logs['actor_loss']:.3f} "
                       f"Alpha={logs['alpha']:.3f} "
                       f"Entropy={logs['entropy']:.3f} "
                      )
                else:
                    tqdm.write(f"[step {step}] Collecting data... (No update yet)")
                            
            # checkpoint
            if step > c_threshold and n_checkpoints>0:
                date = datetime.now().strftime("%Y_%m_%d")
                time = datetime.now().strftime("%H_%M_%S")
                outdir_ckpt = os.path.join(checkpoint_dir, date)
                os.makedirs(outdir_ckpt, exist_ok=True)
                ckpt_path = os.path.join(outdir_ckpt, f"sac_step_{step}_{time}.pt")
                self.save(ckpt_path)
                tqdm.write(f"[SAC] saved checkpoint at step {step}")
                c_threshold+=threshold

        env.close()
        
        #csv saving
        save_training_csv(history,final_path_csv)
        self.save(final_path)
        print(f"[SAC] saved final checkpoint: {final_path}")

    def eval(self, seed, n_episodes: int = 10, path: str = None, render_mode = None):    
        if path is not None:
            self.load(path)
        else:
            print("No checkpoint provided.")
            return -1

        env = make_env_eval(env_id=self.env_id, seed=seed, render_mode=render_mode)
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
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done):
        self.obs[self.ptr] = obs
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
        assert n > 1, "ReplayBuffer needs at least 2 frames to sample"
        idx = np.random.randint(0, n, size=batch_size)
        head_idx = (self.ptr - 1) % self.size
        mask = (idx == head_idx)
        while np.any(mask):
            idx[mask] = np.random.randint(0, n, size=np.sum(mask))
            mask = (idx == head_idx)

        next_idx = (idx + 1) % self.size

        obs = torch.from_numpy(self.obs[idx]).to(self.device).float() / 255.0
        next_obs = torch.from_numpy(self.obs[next_idx]).to(self.device).float() / 255.0

        if obs.shape[-1] == 4:
            obs = obs.permute(0, 3, 1, 2)
            next_obs = next_obs.permute(0, 3, 1, 2)

        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)

        return obs, actions, rewards, next_obs, dones
