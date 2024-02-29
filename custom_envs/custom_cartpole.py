import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, render_mode = "human"):
        super(CustomCartPoleEnv, self).__init__()
        self.length = np.random.uniform(0.3, 0.7)
        self.render_mode = render_mode
        self.total_reward = 0

    def reset(self, **kwargs):
        self.length = np.random.uniform(0.3, 0.7)
        self.total_reward = 0
        observation = super().reset(**kwargs)
        return observation
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.total_reward += reward
        if self.total_reward >= 500:
            self.total_reward = 0
            truncated = True

        return observation, reward, terminated, truncated, info
    
    def render(self):
        super().render()

    def get_length(self):
        return self.length

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_cartpole_env:CustomCartPoleEnv',
)