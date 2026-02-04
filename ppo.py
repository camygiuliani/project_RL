from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils import load_config, save_training_csv
from wrappers import make_env , make_vec_env, make_env_eval


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
    def __init__(self, obs_shape, n_actions, env_id="ALE/SpaceInvaders-v5", n_envs=4, 
                 device=None, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, rollout_len=128):
        
        self.env_id = env_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_len = rollout_len
        self.obs_shape = obs_shape  # (84,84,4)
        self.n_actions = n_actions
        self.n_envs = n_envs

        self.net = ActorCriticCNN(in_channels=4, n_actions=self.n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.buffer = _RolloutBuffer(
            obs_shape=self.obs_shape,
            rollout_len=self.rollout_len,
            n_envs=self.n_envs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        #np.random.seed(self.seed)
        #torch.manual_seed(self.seed)

    """ def preprocess_obs(self, obs_uint8):
        x = torch.from_numpy(obs_uint8).to(self.device).float() / 255.0
        if x.ndim == 3:  # single observation
            x = x.unsqueeze(0)
        # Permute ONLY if looks like NHWC (channels last)
        if x.ndim == 4 and x.shape[-1] in (1, 4) and x.shape[1] not in (1, 4):
             x = x.permute(0, 3, 1, 2)
        return x """
    
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

        # Batch obs: NCHW or NHWC
        elif x.ndim == 4:
            if x.shape[1] in (1, 4):                 # NCHW
                pass
            elif x.shape[-1] in (1, 4):              # NHWC
                x = x.permute(0, 3, 1, 2)            # NCHW
            else:
                raise ValueError(f"Unknown batched obs shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unexpected obs ndim={x.ndim}, shape={tuple(x.shape)}")

        return x / 255.0

    
    """ @torch.no_grad()
    def act(self, obs_uint8):     
        x = self.preprocess_obs(obs_uint8)

        print("Input shape to net:", x.shape)
        logits, values = self.net(x)
        # ... rest of your code

        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.sample()
        logps = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            logps.cpu().numpy(),
            values.cpu().numpy()
        ) """
    
    @torch.no_grad()
    def act(self, obs_uint8):
        # Se per qualche motivo arriva una batch, prendi la prima
        x = self.preprocess_obs(obs_uint8)   # ora deve diventare (1,4,84,84)
        #print("Input shape to net:", x.shape)

        logits, values = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logps = dist.log_prob(actions)
        return (
            actions.cpu().numpy().astype("int64"),
            logps.cpu().numpy(),
            values.cpu().numpy()
        ) 



    @torch.no_grad()
    def value(self, obs_uint8):         
        x = self.preprocess_obs(obs_uint8)
            
        _, v = self.net(x)
        return v.cpu().numpy()

    def update(self, update_epochs=None,  batch_size=None, max_grad_norm=None, 
               vf_coef=None, ent_coef=None, clip_eps=None):
        # to torch
        losses = []
        obs_t = torch.from_numpy(self.buffer.obs).to(self.device).float() / 255.0
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

                losses.append((loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()))

        l, a, c, e = map(np.mean, zip(*losses))
        return {
            "loss": float(l),
            "actor_loss": float(a),
            "critic_loss": float(c),
            "entropy": float(e)
        }

    def train(self,total_steps=1_000_000, n_envs= 4, n_checkpoints=10,update_epochs=4,
                batch_size=256,max_grad_norm=0.5,vf_coef=0.5,ent_coef=0.01,
                clip_eps=0.1, log_every=10_000, checkpoint_dir=None, save_dir=None,):
        
        print("Obs shape:", self.obs_shape)
        seed = 0
        date = datetime.now().strftime("%Y_%m_%d")

        os.makedirs(save_dir, exist_ok=True)
        outdir_runs = os.path.join(save_dir, date)
        os.makedirs(outdir_runs, exist_ok=True)

        final_path = os.path.join(outdir_runs, f"ppo_{total_steps}.pt")
        final_path_csv = os.path.join(outdir_runs, f"metrics_train_{total_steps}.csv")

        #env = make_env(env_id=self.env_id, seed=seed)
        #obs, _ = env.reset(seed=seed)

        env = make_vec_env(self.env_id, n_envs, seed)
        obs, _ = env.reset()

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
                    actions, logps, values = self.act(obs)
                    next_obs, rewards, terms, truncs, _ = env.step(actions)
                    #print("Obs shape:", next_obs.shape)
                    dones = np.logical_or(terms, truncs)

                    #obs=np.transpose(obs, (0, 2, 3, 1))  # NCHW -> NHWC
                    self.buffer.add(obs, actions, rewards, dones, values, logps)
                    obs = next_obs
                    env_steps += self.n_envs
                    pbar.update(self.n_envs)

                    if  dones.any():
                        episode += 1
                        returns_window.append(ep_ret)

                        history.append({
                            "env_step": env_steps,
                            "episode": episode,
                            "ep_len": ep_len,
                            "episodic_return": ep_ret,
                            "avg_return_100": np.mean(returns_window[-100:])
                        })

                        #obs, _ = env.reset()
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

        env = make_env_eval(env_id=self.env_id, seed=seed, render_mode=render_mode)
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            R = 0.0
            while not done:
                obs_batch = np.expand_dims(obs, axis=0)
                action_batch, _, _ = self.act(obs_batch)
                a = action_batch[0]
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
    def __init__(self, obs_shape, rollout_len, n_envs, gamma=0.99, gae_lambda=0.95):
        self.obs_shape = obs_shape
        self.rollout_len = rollout_len
        self.n_envs = n_envs
        self.size = rollout_len * n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        H, W, C = self.obs_shape
        self.obs = np.zeros((self.size, H, W, C), dtype=np.uint8)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.logps = np.zeros(self.size, dtype=np.float32)

        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, actions, rewards, dones, values, logps):
        n = obs.shape[0]
        idx = slice(self.ptr, self.ptr + n)

        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.values[idx] = values
        self.logps[idx] = logps

        self.ptr += n

    def compute_returns_and_advantages(self, last_values):
        adv = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_len)):
            idx = slice(t*self.n_envs, (t+1)*self.n_envs)

            if t == self.rollout_len - 1:
                next_values = last_values
            else:
                next_idx = slice(
                    (t + 1) * self.n_envs,
                    (t + 2) * self.n_envs
                )
                next_values = self.values[next_idx]

            delta = (
                self.rewards[idx]
                + self.gamma * next_values * (1.0 - self.dones[idx])
                - self.values[idx]
            )

            adv = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[idx]) * adv
            self.advantages[idx] = adv

        self.returns = self.advantages + self.values

        # normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / \
                          (self.advantages.std() + 1e-8)
