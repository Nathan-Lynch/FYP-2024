import gymnasium as gym
import optuna
import os
import pickle
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from utils import create_objective

import sys
sys.path.append("C:/Users/natha/Desktop/FYP-2024/")
#sys.path.append("/home/nl6/FYP/FYP-2024")
from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1, CustomCartPoleEnvV2

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV0',
)
gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV1',
    )
gym.envs.register(
    id='CustomCartPole-v2',
    entry_point='custom_cartpole:CustomCartPoleEnvV2',
)

trained_dir = "C:/Users/natha/Desktop/FYP-2024/trained_models"
logdir = "C:/Users/natha/Desktop/FYP-2024/logs/"

#trained_dir = "/home/nl6/FYP/FYP-2024/trained_models"
#logdir = '/home/nl6/FYP/FYP-2024/logs/cartpole/'

model_constant_dir = "cartpole_constant_lr"
model_linear_dir = "cartpole_linear_lr"
model_exponential_dir = "cartpole_exponential_lr"
model_adaptive_dir = "cartpole_adaptive_lr"

cartpole_envs = ["CartPole-v1", "CustomCartPole-v0", "CustomCartPole-v1", "CustomCartPole-v2"]
cartpole_dirs = ["cartpole_constant_lr", "cartpole_linear_lr", "cartpole_exponential_lr", "cartpole_adaptive_lr"]
lr_strategies = ["constant", "linear", "exponential", "adaptive"]

cartpole_timesteps = 250000
cartpole_min_lr = 0.0001 # 1 order of magnitude less than default
cartpole_max_lr = 0.01 # 1 order of magnitude greater than default

def train_model(env_name, timesteps, model, lr_schedule, min_lr, max_lr):
    vec_env = make_vec_env(env_name, n_envs=1)
    eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, env_name, lr_schedule + '_best'),
                        log_path=logdir + env_name, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

    callbacks = [eval_cb]

    obj = create_objective(env_name, model, timesteps, logdir + env_name, callbacks, lr_schedule, min_lr, max_lr)
    study = optuna.create_study(study_name = env_name + lr_schedule, direction = "maximize")
    study.optimize(obj, n_trials = 10)

    with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/" + lr_schedule + "_lr_" + env_name + ".pkl", "wb") as fout:
        pickle.dump(study, fout)

# loop for cartpole experiments, trains new model for each env and lr strategy
for env in cartpole_envs:
    for strategy in  lr_strategies:
        train_model(env, cartpole_timesteps, DQN, strategy, cartpole_min_lr, cartpole_max_lr)