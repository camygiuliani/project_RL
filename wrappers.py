import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import ale_py

class AtariPreprocess(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        c = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, c), dtype=np.uint8
        )

    def observation(self, obs):
        # obs: (H,W,3) uint8
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H,W)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = obs[..., None]  # (H,W,1)
        return obs.astype(np.uint8)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

def make_env(env_id="ALE/SpaceInvaders-v5", seed=0, frame_skip=4,render_mode=None):
    env = gym.make(env_id, frameskip=frame_skip, render_mode=render_mode)
    env.reset(seed=seed)
    env = AtariPreprocess(env, width=84, height=84, grayscale=True)
    env = FrameStack(env, k=4)
    return env
