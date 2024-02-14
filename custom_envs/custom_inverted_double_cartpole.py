import gymnasium as gym
from gymnasium.envs.mujoco import InvertedDoublePendulumEnv
import numpy as np

class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    metadata = {
        "render_modes": [
        "human",
        "rgb_array",
        "depth_array",
        ],
    }
    def __init__(self, render_mode = "human"):
        super(CustomInvertedDoublePendulumEnv, self).__init__()
        self.render_mode = render_mode
        self.min_pole_len = 0.1
        self.max_pole_len = 0.5
        self.xml_file = "inverted_double_pendulum.xml"

    def reset(self, **kwargs):
        pole_length1 = np.random.uniform(self.min_pole_length, self.max_pole_length)
        pole_length2 = np.random.uniform(self.min_pole_length, self.max_pole_length)

        self._modify_xml(pole_length1, pole_length2)

        observation = super().reset(**kwargs)
        return observation

    def _modify_xml(self, pole_length1, pole_length2):
        with open(self.xml_file, "r") as file:
            xml_content = file.read()

        xml_content = xml_content.replace('size="0.045 0.3"', f'size="0.045 {pole_length1}"')
        xml_content = xml_content.replace('size="0.045 0.3"', f'size="0.045 {pole_length2}"')

        with open("inverted_double_pendulum.xml", "w") as file:
            file.write(xml_content)

        self.model = self.sim.load_model_from_xml(xml_content)
    
    def render(self):
        super().render()

gym.envs.register(
    id='CustomInvertedDoublePendulum-v0',
    entry_point='custom_inverted_double_cartpole:CustomInvertedDoublePendulumEnv',
)