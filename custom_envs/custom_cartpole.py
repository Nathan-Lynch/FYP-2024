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

    def reset(self, **kwargs):
        self.length = np.random.uniform(0.3, 7)
        observation = super().reset(**kwargs)
        return observation
    
    def render(self):
        super().render()

    def get_length(self):
        return self.length

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_cartpole_env:CustomCartPoleEnv',
)