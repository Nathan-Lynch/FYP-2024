import gymnasium as gym
from stable_baselines3 import PPO

model_dir = "trained_BW_PPO_model"
logdir = "logs"

TIMESTEPS = 10_000_000

env = gym.make("BipedalWalker-v3", render_mode=None)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name="BW_PPO")
model.save('C:/Users/natha/Desktop/FYP-2024/trained_models/{}'.format(model_dir))