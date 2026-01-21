import numpy as np
from tqdm import tqdm
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import ale_py 
from wrappers import make_env



class DQNCNN(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    
        self.fc1 = nn.Linear(7 * 7 * 64, 512) 
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def linear_eps(step, eps_start=1.0, eps_end=0.1, decay_steps=1_000_000):
    t = min(step / decay_steps, 1.0)
    return eps_start + t * (eps_end - eps_start)

class DQN_Agent:
    def __init__(self,env: str,n_channels: int, n_actions: int, device: torch.device, gamma=0.99, lr=1e-4,
                 double_dqn=True):

        self.env = env
        self.n_channels = n_channels
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.double_dqn = double_dqn
        
        #1.1 Initialize action-value function Q with random weights
        self.q = DQNCNN(self.n_actions, self.n_channels).to(self.device)
        #1.2 Initialize target action-value function Q with the same weights
        self.tgt = DQNCNN(self.n_actions, self.n_channels).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        
        self.tgt.eval()
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_arr, eps: float):
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)

        # obs: (84,84,4) uint8 -> (1,4,84,84) float
        obs_arr = np.array(obs_arr)
        x = torch.from_numpy(obs_arr).to(self.device)
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

        #6 set y_j
        with torch.no_grad():
            if self.double_dqn:
            #DDQN
                # action selection from online net
                next_actions = torch.argmax(self.q(next_obs_t), dim=1)  # (B,)
                # value from target net
                next_q = self.tgt(next_obs_t).gather(1, next_actions.view(-1,1)).squeeze(1)

            else:
            #DQN
                next_q = torch.max(self.tgt(next_obs_t), dim=1).values

            #Bellman Equation  
            target = rews_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.smooth_l1_loss(q_a, target)
        self.optim.zero_grad(set_to_none=True)

        #7 Gradient descent step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        return float(loss.item())
    
    def train(self, batch_size:int, buffer_size:int, total_steps:int,
               l_start:int, train_f:int, target_update:int, n_checkpoints:int,
               save_dir: str = "runs/dqn"):
        
        seed = 0  
        threshold = total_steps/n_checkpoints 
        c_threshold = threshold
        final_path = os.path.join(save_dir, f"dqn_{total_steps}.pt")

        print(f"Using device: {self.device}")
        env = make_env(env_id=self.env, seed=seed)
        obs_shape = env.observation_space.shape  # (84,84,4)

        #1.3  Initialize replay memory
        rb = ReplayBuffer(buffer_size, obs_shape)

        obs, _ = env.reset(seed=seed)
        ep_ret = 0.0
        ep_len = 0
        episode = 0

        os.makedirs("checkpoints", exist_ok=True)

        pbar = tqdm(range(1, total_steps + 1))
        for step in pbar:
            eps = linear_eps(step)
            #2 Random action selection
            action = self.act(obs, eps)

            #3 Execute action a in emulator and observe reward r and image x'
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            #4 Store transition (x,a,r,x',done) in replay memory
            rb.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if done:
                episode += 1
                obs, _ = env.reset()
                pbar.set_description(f"ep={episode} R={ep_ret:.1f} len={ep_len} eps={eps:.2f} rb={len(rb)}")
                ep_ret, ep_len = 0.0, 0

            # learning
            if step > l_start and step % train_f == 0:
                #5 Sample random minibatch of transitions (x,a,r,x',done) from replay memory
                batch = rb.sample(batch_size)
                loss = self.update(batch)

            # target update
            if step % target_update == 0:
                self.sync_target()

            # checkpoint
            if step > c_threshold:
                ckpt_path = f"checkpoints/dqn_step_{step}.pt"
                self.save(ckpt_path)
                c_threshold+=threshold
        env.close()

        #final save
        self.save(final_path)

    def eval(self, seed, n_episodes: int = 10, path: str = None):    
        if path is not None:
            self.load(path)
        else:
            print("No checkpoint provided.")
            return -1

        env = make_env(env_id=self.env, seed=seed)
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            R = 0.0
            while not done:
                a = self.act(obs, eps=0.0)
                obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                R += r
            returns.append(R)
            print(f"Episode {ep+1}: return={R:.1f}")

        mean_return = float(np.mean(returns))
        print("Mean return:", mean_return)
        env.close()
        return mean_return
    
    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.sync_target()



class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, device=None):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.acts = np.zeros((capacity,), dtype=np.int64)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, o, a, r, no, done):
        self.obs[self.idx] = o
        self.next_obs[self.idx] = no
        self.acts[self.idx] = a
        self.rews[self.idx] = r
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idxs], self.acts[idxs], self.rews[idxs],
                self.next_obs[idxs], self.dones[idxs])

    def __len__(self):
        return self.size
