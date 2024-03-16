import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np

class CustomCartPoleEnvV0(CartPoleEnv):
    '''
    Custom CartPole environment
    Reward is changed to reflect the Pole angle, normalized to be in range [0,1]
    Reward = 1 - abs(current pole angle - max pole angle before termination)
    '''
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, render_mode = "human"):
        super(CustomCartPoleEnvV0, self).__init__()
        self.max_pole_angle = 0.2095
        self.render_mode = render_mode
        self.total_reward = 0

    def reset(self, **kwargs):
        self.total_reward = 0
        observation = super().reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # observation = [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        pole_angle = observation[2]

        if terminated or truncated:
            reward = 0
        else:
            reward = reward - abs(pole_angle/self.max_pole_angle)
        self.total_reward += reward

        if self.total_reward >= 500:
            self.total_reward = 0
            truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        super().render()

class CustomCartPoleEnvV1(CustomCartPoleEnvV0):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode = "human"):
        super(CustomCartPoleEnvV1, self).__init__()
        self.length = 0.3
        self.max_length = 0.8
        self.render_mode = render_mode
        self.total_reward = 0

    def reset(self, **kwargs):
        if self.length < self.max_length:
            self.length += 0.0001
        else:
            self.length = self.max_length
        self.total_reward = 0
        observation = super().reset(**kwargs)
        return observation

    def render(self):
        super().render()

class CustomCartPoleEnvV2(CustomCartPoleEnvV0):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode = "human"):
        super(CustomCartPoleEnvV2, self).__init__()
        self.length = np.random.uniform(0.3, 0.7)
        self.render_mode = render_mode
        self.total_reward = 0

    def reset(self, **kwargs):
        self.length = np.random.uniform(0.3, 0.7)
        self.total_reward = 0
        observation = super().reset(**kwargs)
        return observation

    def render(self):
        super().render()

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_cartpole:CustomCartPoleEnvV0',
)
gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='custom_cartpole:CustomCartPoleEnvV1',
)
gym.envs.register(
    id='CustomCartPole-v2',
    entry_point='custom_cartpole:CustomCartPoleEnvV2',
)