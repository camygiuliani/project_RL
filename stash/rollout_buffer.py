# rollout_buffer_ppo.py
import numpy as np


class RolloutBuffer:
    """
    On-policy buffer for PPO with GAE(lambda).
    Stores uint8 observations (H,W,C) to save memory.
    """

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
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.logps[self.ptr] = logp

        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def compute_returns_and_advantages(self, last_value):
        """
        GAE(lambda):
          delta_t = r_t + gamma * V_{t+1} * (1-done) - V_t
          A_t = delta_t + gamma*lambda*(1-done) * A_{t+1}
          R_t = A_t + V_t
        """
        assert self.full, "Buffer not full yet"
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv

        self.returns = self.advantages + self.values

        # normalize advantages (standard PPO trick)
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_minibatches(self, batch_size, rng=None):
        assert self.full, "Buffer not full yet"
        idx = np.arange(self.size)
        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)

        for start in range(0, self.size, batch_size):
            mb = idx[start:start + batch_size]
            yield (
                self.obs[mb],
                self.actions[mb],
                self.logps[mb],
                self.advantages[mb],
                self.returns[mb],
            )
