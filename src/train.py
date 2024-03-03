import gymnasium as gym
import numpy as np
import optuna
import os
import pickle
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from utils import linear_schedule, exponential_schedule, create_objective, AdaptiveLearningRateScheduler
from custom_callbacks import AdaptiveLRCallback

import sys
#sys.path.append("/home/nl6/FYP/FYP-2024/")
sys.path.append("C:/Users/35385/Desktop/FYP-2024")
#from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1



# For each want environment, want constant, linear, exponential, adaptive learning rates

trained_dir = "C:/Users/natha/Desktop/FYP-2024/trained_models"
logdir = 'C:/Users/natha/Desktop/FYP-2024/logs/cartpole/'
model_constant_dir = "cartpole_constant_lr"
model_linear_dir = "cartpole_linear_lr"

# Default cartpole

env = gym.make("CartPole-v1", render_mode = None)
vec_env = make_vec_env("CartPole-v1", n_envs = 16)

cartpole_timesteps = 250000
reward_cb = StopTrainingOnRewardThreshold(reward_threshold = 500, verbose = 1) # stays the same for all cartpole environments

# constant lr
eval_cb = EvalCallback(env, best_model_save_path=os.path.join(trained_dir, model_constant_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False, callback_after_eval=reward_cb)

callbacks = CallbackList([eval_cb])

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "constant", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 3)

with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/constant_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# linear decreasing lr
eval_cb = EvalCallback(env, best_model_save_path=os.path.join(trained_dir, model_linear_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False, callback_after_eval=reward_cb)

callbacks = CallbackList([eval_cb])

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "linear", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 3)

with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/linear_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)


# CustomCartpole-v0

# CustomCartpole-v1