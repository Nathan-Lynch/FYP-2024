import gymnasium as gym
from custom_envs.custom_cartpole import CustomCartPoleEnv
from custom_envs.custom_inverted_double_cartpole import CustomInvertedDoublePendulumEnv

'''gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnv',
    )

env = gym.make("CustomCartPole-v0", render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env = gym.make("CustomCartPole-v0", render_mode="human")
        observation, info = env.reset()

env.close()'''

gym.envs.register(
    id='CustomInvertedDoublePendulum-v0',
    entry_point='custom_envs.custom_inverted_double_cartpole:CustomInvertedDoublePendulumEnv',
)

env = gym.make("CustomInvertedDoublePendulum-v0", render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env = gym.make("CustomInvertedDoublePendulum-v0", render_mode="human")
        observation, info = env.reset()

env.close()