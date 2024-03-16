import gymnasium as gym
from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv
import mujoco
import numpy as np

class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, render_mode="human"):
        self.min_pole_len = 0.3  # Minimum pole length
        self.max_pole_len = 0.8  # Maximum pole length
        super().__init__(render_mode=render_mode)
        self.pole1_idx = 3
        self.pole2_idx = 4
        self.pole_idx = 1

    def reset(self, **kwargs):
        pole_length1 = np.random.uniform(self.min_pole_len, self.max_pole_len)
        pole_length2 = np.random.uniform(self.min_pole_len, self.max_pole_len)
        
        # Update pole lengths in the model
        self.update_pole_lengths(pole_length1, pole_length2)
        
        return super().reset(**kwargs)

    def update_pole_lengths(self, pole_length1, pole_length2):
        self.model.geom_size[self.pole1_idx, self.pole_idx] = pole_length1
        self.model.geom_size[self.pole2_idx, self.pole_idx] = pole_length2

        # Force the model to recompute derived values after the modification
        mujoco.mj_forward(self.model, self.data)

gym.envs.register(
    id='CustomInvertedDoublePendulum-v0',
    entry_point='custom_inverted_double_cartpole:CustomInvertedDoublePendulumEnv',
)