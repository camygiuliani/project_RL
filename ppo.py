from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils import load_config, save_training_csv
from wrappers import make_env


class ActorCriticCNN(nn.Module):
    """Shared CNN encoder -> actor logits + critic value. Input: (N,4,84,84) float in [0,1]."""
    def __init__(self, in_channels, n_actions):
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
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


class PPO_Agent:
    def __init__(self, obs_shape, n_actions, env_id="ALE/SpaceInvaders-v5", device=None, 
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95, rollout_len=128):
        
        self.env_id = env_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_len = rollout_len
        self.obs_shape = obs_shape  # (84,84,4)
        self.n_actions = n_actions

        self.net = ActorCriticCNN(in_channels=self.obs_shape[2], n_actions=self.n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.buffer = _RolloutBuffer(
            obs_shape=self.obs_shape,
            size=self.rollout_len,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        #np.random.seed(self.seed)
        #torch.manual_seed(self.seed)

    @torch.no_grad()
    def act(self, obs_uint8):
        x = torch.from_numpy(obs_uint8).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1,4,84,84)
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs_uint8):
        x = torch.from_numpy(obs_uint8).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        _, v = self.net(x)
        return float(v.item())

    def update(self, update_epochs=None,  batch_size=None, max_grad_norm=None, 
               vf_coef=None, ent_coef=None, clip_eps=None):
        # to torch
        obs_t = torch.from_numpy(self.buffer.obs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        actions_t = torch.from_numpy(self.buffer.actions).to(self.device)
        old_logps_t = torch.from_numpy(self.buffer.logps).to(self.device)
        adv_t = torch.from_numpy(self.buffer.advantages).to(self.device)
        ret_t = torch.from_numpy(self.buffer.returns).to(self.device)

        N = obs_t.shape[0]
        idx = np.arange(N)

        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, batch_size):
                mb = idx[start:start + batch_size]

                logits, values = self.net(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logps = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logps - old_logps_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, ret_t[mb])

                loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                self.opt.step()

                return {
                    "loss": float(loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy.item())
                }

    def train(
    self,
    total_steps=1_000_000,
    n_checkpoints=10,
    update_epochs=4,
    batch_size=256,
    max_grad_norm=0.5,
    vf_coef=0.5,
    ent_coef=0.01,
    clip_eps=0.1,
    log_every=10_000,
    checkpoint_dir=None,
    save_dir=None,):
        
        seed = 0
        date = datetime.now().strftime("%Y_%m_%d")

        os.makedirs(save_dir, exist_ok=True)
        outdir_runs = os.path.join(save_dir, date)
        os.makedirs(outdir_runs, exist_ok=True)

        final_path = os.path.join(outdir_runs, f"ppo_{total_steps}.pt")
        final_path_csv = os.path.join(outdir_runs, f"metrics_train_{total_steps}.csv")

        env = make_env(env_id=self.env_id, seed=seed)
        obs, _ = env.reset(seed=seed)

        history = []
        returns_window = []

        ep_ret = 0.0
        ep_len = 0
        episode = 0

        env_steps = 0
        num_updates = total_steps // self.rollout_len
        checkpoint_every = total_steps // n_checkpoints if n_checkpoints > 0 else None

        tqdm.write("Starting PPO training...")

        with tqdm(total=total_steps, desc="PPO Training") as pbar:
            for update in range(num_updates):

                # ---------- LR ANNEALING ----------
                frac = 1.0 - (update / num_updates)
                lr_now = self.lr * frac
                for pg in self.opt.param_groups:
                    pg["lr"] = lr_now

                # ---------- COLLECT ROLLOUT ----------
                self.buffer.reset()

                for step in range(self.rollout_len):
                    action, logp, value = self.act(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    self.buffer.add(obs, action, reward, done, value, logp)

                    ep_ret += reward
                    ep_len += 1
                    env_steps += 1
                    pbar.update(1)

                    obs = next_obs

                    if done:
                        episode += 1
                        returns_window.append(ep_ret)

                        history.append({
                            "env_step": env_steps,
                            "episode": episode,
                            "ep_len": ep_len,
                            "episodic_return": ep_ret,
                            "avg_return_100": np.mean(returns_window[-100:])
                        })

                        obs, _ = env.reset()
                        ep_ret = 0.0
                        ep_len = 0

                    if env_steps >= total_steps:
                        break

                # ---------- GAE ----------
                with torch.no_grad():
                    last_value = self.value(obs)

                self.buffer.compute_returns_and_advantages(last_value)

                # ---------- PPO UPDATE ----------
                self.net.train()
                logs = self.update(
                    update_epochs=update_epochs,
                    batch_size=batch_size,
                    max_grad_norm=max_grad_norm,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    clip_eps=clip_eps,
                )

                # ---------- LOGGING ----------
                if env_steps % log_every < self.rollout_len:
                    avg100 = np.mean(returns_window[-100:]) if returns_window else 0.0
                    tqdm.write(
                        f"[Steps {env_steps}] "
                        f"Avg100={avg100:.1f} "
                        f"Loss={logs['loss']:.3f} "
                        f"Actor={logs['actor_loss']:.3f} "
                        f"Critic={logs['critic_loss']:.3f} "
                        f"Ent={logs['entropy']:.3f} "
                        f"LR={lr_now:.6f}"
                    )

                # ---------- CHECKPOINT ----------
                if checkpoint_every and env_steps >= checkpoint_every:
                    ckpt_dir = os.path.join(checkpoint_dir, date)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"ppo_step_{env_steps}.pt")
                    self.save(ckpt_path)
                    tqdm.write(f"Checkpoint saved at {env_steps} steps")
                    checkpoint_every += total_steps // n_checkpoints

                if env_steps >= total_steps:
                    break

        env.close()
        self.save(final_path)
        save_training_csv(history, final_path_csv)

        print(f"Training finished. Saved final model to {final_path}")
        return final_path
    
    
    def eval(self, seed=0, n_episodes: int = 10, path: str = None, render_mode=None):    
        if path is not None:
            print(f"Loading checkpoint from: {path}")
            if not os.path.exists(path):
                print(f"Checkpoint not found: {path}")
                return -1
            self.load(path)
        else:
            print("Using current model weights for evaluation.")

        env = make_env(env_id=self.env_id, seed=seed, render_mode=render_mode)
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            R = 0.0
            while not done:
                a,_,_ = self.act(obs)
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

    
    def save(self, path):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.net.to(self.device)
        self.net.eval()

class _RolloutBuffer:
    """On-policy rollout buffer + GAE(lambda). Stores obs as uint8 (H,W,C)."""

    def __init__(self, obs_shape, size, gamma=0.99, gae_lambda=0.95):
        self.obs_shape = obs_shape
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        H, W, C = self.obs_shape
        self.obs = np.zeros((self.size, H, W, C), dtype=np.uint8)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)
        self.values = np.zeros((self.size,), dtype=np.float32)
        self.logps = np.zeros((self.size,), dtype=np.float32)

        self.advantages = np.zeros((self.size,), dtype=np.float32)
        self.returns = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, logp):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = float(value)
        self.logps[self.ptr] = float(logp)

        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def compute_returns_and_advantages(self, last_value):
        assert self.full, "Rollout buffer not full"
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv

        self.returns = self.advantages + self.values

        # normalize advantages
        m = self.advantages.mean()
        s = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - m) / s

