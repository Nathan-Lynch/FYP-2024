import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import RewardLossCallback
import os
import numpy as np

base_dir = 'C:/Users/35385/Desktop/FYP-2024/trained_models'
model_dir = "trained_BW_PPO_model"
logdir = "logs"

TIMESTEPS = 10_000_000

env = gym.make("BipedalWalker-v3", render_mode=None)

def train_model(algorithm, env, model_dir, tb_log_name):

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    reward_loss_callback = RewardLossCallback()

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'),
                                 log_path=logdir, eval_freq=1000, n_eval_episodes=10,
                                 deterministic=True, render=False, callback_after_eval=reward_threshold_callback)

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)

    callback = CallbackList([eval_callback, reward_loss_callback])

    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

train_model(PPO, env, model_dir, "bw_ppo")