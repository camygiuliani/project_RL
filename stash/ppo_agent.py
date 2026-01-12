# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCNN(nn.Module):
    """
    Shared CNN encoder -> 2 heads:
      - actor: logits over discrete actions
      - critic: scalar value V(s)
    Input: (N,4,84,84) float in [0,1]
    """
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # 84x84 with Atari convs -> 7x7x64 = 3136
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


class PPOAgent:
    def __init__(
        self,
        in_channels,
        n_actions,
        device,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCriticCNN(in_channels, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_uint8):
        """
        obs_uint8: (84,84,4) uint8
        returns: action(int), logp(float), value(float)
        """
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

    def update(self, obs_uint8, actions, old_logps, advantages, returns, n_epochs=4, batch_size=256):
        """
        PPO-Clip update.
        obs_uint8: (N,84,84,4) uint8
        actions: (N,)
        old_logps: (N,)
        advantages: (N,)
        returns: (N,)
        """
        N = obs_uint8.shape[0]

        # to torch
        obs_t = torch.from_numpy(obs_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        actions_t = torch.from_numpy(actions).to(self.device)
        old_logps_t = torch.from_numpy(old_logps).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        idx = np.arange(N)

        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, batch_size):
                mb = idx[start:start + batch_size]

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
