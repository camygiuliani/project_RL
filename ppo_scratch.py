# ppo_scratch.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers import make_env


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


class PPOTrainer:
    """
    PPO-Clip from scratch (discrete actions) designed to plug into your project.
    Use:
        trainer = PPOTrainer(env_id="ALE/SpaceInvaders-v5", seed=0)
        trainer.train(total_steps=2_000_000)
    """

    def __init__(
        self,
        env_id="ALE/SpaceInvaders-v5",
        seed=0,
        device=None,
        # PPO hyperparams
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # rollout/update
        rollout_len=128,
        n_epochs=4,
        batch_size=256,
        # logging/saving
        save_dir="runs/ppo_scratch",
        eval_every=100_000,
        eval_episodes=10,
    ):
        self.env_id = env_id
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.save_dir = save_dir
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes

        os.makedirs(self.save_dir, exist_ok=True)

        # env to infer shapes
        env = make_env(env_id=self.env_id, seed=self.seed)
        self.obs_shape = env.observation_space.shape  # (84,84,4)
        self.n_actions = env.action_space.n
        env.close()

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
    def _act(self, obs_uint8):
        x = torch.from_numpy(obs_uint8).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1,4,84,84)
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def _value(self, obs_uint8):
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

    def save(self, path):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.net.to(self.device)
        self.net.eval()

    def _update(self):
        # to torch
        obs_t = torch.from_numpy(self.buffer.obs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        actions_t = torch.from_numpy(self.buffer.actions).to(self.device)
        old_logps_t = torch.from_numpy(self.buffer.logps).to(self.device)
        adv_t = torch.from_numpy(self.buffer.advantages).to(self.device)
        ret_t = torch.from_numpy(self.buffer.returns).to(self.device)

        N = obs_t.shape[0]
        idx = np.arange(N)

        for _ in range(self.n_epochs):
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

    def train(self, total_steps=2_000_000):
        env = make_env(env_id=self.env_id, seed=self.seed)
        obs, _ = env.reset(seed=self.seed)

        steps = 0
        next_eval = self.eval_every

        while steps < total_steps:
            self.buffer.reset()

            # collect rollout
            for _ in range(self.rollout_len):
                action, logp, value = self._act(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                self.buffer.add(obs, action, reward, done, value, logp)

                obs = next_obs
                steps += 1

                if done:
                    obs, _ = env.reset()

                if steps >= total_steps:
                    break

            # bootstrap value for last state
            last_v = self._value(obs)
            self.buffer.full = True
            self.buffer.compute_returns_and_advantages(last_value=last_v)

            # update
            self.net.train()
            self._update()
            self.net.eval()

            # eval + checkpoint
            if steps >= next_eval:
                mean_r, std_r = self.eval()
                print(f"[PPO scratch] step={steps} eval_mean={mean_r:.2f} eval_std={std_r:.2f}")
                ckpt_path = os.path.join(self.save_dir, f"ppo_scratch_step_{steps}.pt")
                self.save(ckpt_path)
                next_eval += self.eval_every

        env.close()
        final_path = os.path.join(self.save_dir, "ppo_scratch_final.pt")
        self.save(final_path)
        print(f"Saved final PPO scratch checkpoint: {final_path}")
        return final_path
