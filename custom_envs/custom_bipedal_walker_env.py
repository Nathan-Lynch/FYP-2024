import gymnasium as gym
from gymnasium.envs.box2d import BipedalWalker
from typing import Optional
import numpy as np
from Box2D.b2 import fixtureDef, polygonShape

class CustomBipedalWalkerEnv(BipedalWalker):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        }

    def __init__(self, render_mode: Optional[str] = None):
        super(CustomBipedalWalkerEnv, self).__init__()
        self.render_mode = render_mode

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return observation

    def render(self):
        super().render()

gym.envs.register(
    id='CustomBipedalWalker-v0',
    entry_point='custom_bipedal_walker_env:CustomBipedalWalkerEnv',
)