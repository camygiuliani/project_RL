import numpy as np
import random

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
