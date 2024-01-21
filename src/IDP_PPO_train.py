import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import os

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models/'
logdir = "logs"

model1_dir = "trained_IDP_PPO_model"
model2_dir = "trained_IDP_SAC_model"
model3_dir = "trained_IDP_td3_model"

TIMESTEPS = 1_000_000

env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")

def train_model(algorithm, env, model_dir, tb_log_name):
    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name)
    model.save(os.path.join(base_dir, model_dir))

#train_model(PPO, env, model1_dir, "IDP_PPO")
#train_model(SAC, env, model2_dir, "IDP_SAC")
train_model(TD3, env, model3_dir, "IDP_td3")