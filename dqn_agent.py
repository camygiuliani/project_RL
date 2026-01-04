import numpy as np
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(
        self,
        q_net,
        target_net,
        n_actions: int,
        device: torch.device,
        gamma=0.99,
        lr=1e-4,
        double_dqn=True,
    ):
        self.q = q_net.to(device)
        self.tgt = target_net.to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.tgt.eval()

        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.double_dqn = double_dqn

        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_uint8, eps: float):
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)

        # obs: (84,84,4) uint8 -> (1,4,84,84) float
        x = torch.from_numpy(obs_uint8).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        q = self.q(x)
        return int(torch.argmax(q, dim=1).item())

    def update(self, batch):
        obs, acts, rews, next_obs, dones = batch

        obs_t = torch.from_numpy(obs).to(self.device).permute(0,3,1,2).float() / 255.0
        next_obs_t = torch.from_numpy(next_obs).to(self.device).permute(0,3,1,2).float() / 255.0
        acts_t = torch.from_numpy(acts).to(self.device)
        rews_t = torch.from_numpy(rews).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.q(obs_t)  # (B,A)
        q_a = q_values.gather(1, acts_t.view(-1,1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # action selection from online net
                next_actions = torch.argmax(self.q(next_obs_t), dim=1)  # (B,)
                # value from target net
                next_q = self.tgt(next_obs_t).gather(1, next_actions.view(-1,1)).squeeze(1)
            else:
                # vanilla DQN
                next_q = torch.max(self.tgt(next_obs_t), dim=1).values

            target = rews_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.smooth_l1_loss(q_a, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        return float(loss.item())

    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.sync_target()
