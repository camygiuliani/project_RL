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
    def __init__(self, obs_shape, n_actions, env_id="ALE/SpaceInvaders-v5", seed=0, device=None, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.1, 
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, rollout_len=128, update_epochs=4, batch_size=256,
        save_dir="runs/ppo",eval_episodes=10):
        
        self.env_id = env_id
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading yaml config
        self.cfg = load_config("config.yaml")

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.rollout_len = rollout_len
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.eval_episodes = eval_episodes

        os.makedirs(self.save_dir, exist_ok=True)

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

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

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

    @torch.no_grad()
    def eval(self, seed_offset=10_000):
        env = make_env(env_id=self.env_id, seed=self.seed + seed_offset)
        rets = []
        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                x = torch.from_numpy(obs).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                logits, _ = self.net(x)
                action = int(torch.argmax(logits, dim=1).item())  # deterministic eval
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                ep_ret += float(r)
            rets.append(ep_ret)
        env.close()
        return float(np.mean(rets)), float(np.std(rets))
        
    def update(self):
        # to torch
        obs_t = torch.from_numpy(self.buffer.obs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        actions_t = torch.from_numpy(self.buffer.actions).to(self.device)
        old_logps_t = torch.from_numpy(self.buffer.logps).to(self.device)
        adv_t = torch.from_numpy(self.buffer.advantages).to(self.device)
        ret_t = torch.from_numpy(self.buffer.returns).to(self.device)

        N = obs_t.shape[0]
        idx = np.arange(N)

        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]

                logits, values = self.net(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logps = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logps - old_logps_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, ret_t[mb])

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                return {
                    "loss": float(loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy.item())
                }

    def train(self, total_steps=1_000_000, save_dir: str = None, n_checkpoints: int = 10, log_every: int = 1_000):
        # 1. Setup Directories
        date = datetime.now().strftime("%Y_%m_%d")
        outdir_runs = os.path.join(save_dir, date)
        os.makedirs(outdir_runs, exist_ok=True)
        
        final_path = os.path.join(outdir_runs, f"ppo_{total_steps}.pt")
        final_path_csv = os.path.join(outdir_runs, f"metrics_train_{total_steps}.csv")

        # 2. Metrics Setup
        history = []
        returns_window = []
        episode = 0
        ep_ret = 0.0
        ep_len = 0
        
        ep_actor_losses = []
        ep_critic_losses = []
        ep_entropies = []
        ep_total_losses = []

        # 3. Env Setup
        env = make_env(env_id=self.env_id, seed=self.seed)
        obs, _ = env.reset(seed=self.seed)

        # 4. Calculate Updates required
        #    If total_steps=1M and rollout_len=2048, we need ~488 updates.
        num_updates = total_steps // self.rollout_len
        
        g_step = 0
        
        # Checkpoint logic
        threshold = total_steps // n_checkpoints 
        c_threshold = threshold
        l_threshold = log_every

        tqdm.write("Starting Discrete PPO training...")
        
        with tqdm(total=total_steps, desc="PPO Training", leave=True) as pbar:
            while g_step < total_steps:
                self.buffer.reset()
                for _ in range(self.rollout_len):
                    action, logp, value = self.act(obs)
                    next_obs, reward, term, trunc, _ = env.step(action)
                    done = term or trunc

                    ep_ret += float(reward)
                    ep_len += 1

                    self.buffer.add(obs, action, reward, done, value, logp)

                    obs = next_obs
                    
                    if done:
                        episode += 1
                        returns_window.append(ep_ret)
                        avg100 = sum(returns_window[-100:]) / min(len(returns_window), 100)

                        # Calculate episode means
                        actor_mean  = sum(ep_actor_losses)/len(ep_actor_losses) if ep_actor_losses else None
                        critic_mean = sum(ep_critic_losses)/len(ep_critic_losses) if ep_critic_losses else None
                        ent_mean    = sum(ep_entropies)/len(ep_entropies) if ep_entropies else None
                        loss_mean   = sum(ep_total_losses)/len(ep_total_losses) if ep_total_losses else None

                        history.append({
                            "env_step": int(g_step),
                            "episode": int(episode),
                            "ep_len": int(ep_len),
                            "episodic_return": float(ep_ret),
                            "avg_return_100": float(avg100),
                            "actor_loss_ep_mean": actor_mean,
                            "critic_loss_ep_mean": critic_mean,
                            "entropy_ep_mean": ent_mean,
                            "total_loss_ep_mean": loss_mean,
                            "updates_in_episode": int(len(ep_actor_losses)),
                        })

                        # Reset episode vars
                        ep_ret = 0.0
                        ep_len = 0
                        ep_actor_losses.clear()
                        ep_critic_losses.clear()
                        ep_entropies.clear()
                        ep_total_losses.clear()
                        obs, _ = env.reset()

            
                last_v = self.value(obs)
                self.buffer.full = True
                self.buffer.compute_returns_and_advantages(last_value=last_v)

                self.net.train()
                logs = self.update() # PPO update happens here

                if g_step > l_threshold:
                    if logs is not None:
                        tqdm.write(
                            f"[Step {g_step}] "
                            f"Rew={returns_window[-1] if returns_window else 0:.0f} "
                            f"Loss={logs['loss']:.2f} "
                            f"Actor={logs['actor_loss']:.2f} "
                            f"Critic={logs['critic_loss']:.2f} "
                            f"Ent={logs['entropy']:.2f}"
                        )
                    l_threshold += log_every

                if g_step > c_threshold:
                    time = datetime.now().strftime("%H_%M_%S")
                    outdir_ckpt = os.path.join(self.cfg["ppo"]["checkpoints_dir"], date)
                    os.makedirs(outdir_ckpt, exist_ok=True)
                    ckpt_path = os.path.join(outdir_ckpt, f"ppo_step_{g_step}_{time}.pt")
                    self.save(ckpt_path)
                    c_threshold += threshold
                    tqdm.write(f"Checkpoint saved at step {g_step}")

                g_step += 1
                pbar.update(1)



        tqdm.write("Training finished. Saving final models...")
        env.close()
        self.save(final_path)
        save_training_csv(history, final_path_csv)
        print(f"Saved final PPO: {final_path}")
        return final_path
    
    
    def eval(self, seed=0, n_episodes: int = 10, path: str = None, render_mode=None):    
        if path is not None:
            print(f"Loading checkpoint from: {path}")
            if not os.path.exists(path):
                print(f"Submitted checkpoint not found: {path}")
                return -1
            self.load(path)
        else:
            print("No checkpoint provided or path is None.")
            return -1

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

