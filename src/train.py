import gymnasium as gym
import optuna
import os
import pickle
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from optuna.samplers import TPESampler
from utils import create_objective

import sys
sys.path.append("/home/nl6/FYP/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1, CustomCartPoleEnvV2, CustomCartPoleEnvV3
from custom_envs.custom_inverted_double_pendulum import CustomInvertedDoublePendulumEnvV0, CustomInvertedDoublePendulumEnvV1, CustomInvertedDoublePendulumEnvV2

# Registering Custom Environments
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
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV2',
)
gym.envs.register(
    id='CustomCartPole-v3',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV3',
)

# Common between all environments
seed_val = 2002
lr_strategies = ["constant", "linear", "exponential", "adaptive", "adaptivet"]

trained_dir = "/home/nl6/FYP/FYP-2024/trained_models"
logdir = '/home/nl6/FYP/FYP-2024/logs/'

# Maybe put in Utils.py
def train_model(env_name, timesteps, model, lr_schedule, min_lr, max_lr, trials, rl_t):
    '''
    Creates an Optuna study and trains the model on a given environment.
    Saves studies in saved_studies directory.

    :param env_name: Name of the environment.
    :param timesteps: Number of timesteps for training.
    :param model: Model used for training.
    :param lr_schedule: Learning rate strategy to br used.
    :param min_lr: Minimum value allowed for learning rate.
    :param max_lr: Maximum value allowed for learing rate.
    :param trials: Number of trials to run the Study for.
    :param rl_t: Reward Loss threshold for adaptive_t learning rate
    '''
    # Set up environment
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed_val)

    # Set up callbacks to be used
    eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, env_name, lr_schedule + '_best'),
                        log_path=logdir + env_name, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)
    callbacks = [eval_cb]

    # Creates an Optuna study and runs n_trials for the study
    obj = create_objective(env_name, model, timesteps, logdir + env_name, callbacks, lr_schedule, min_lr, max_lr, rl_t)
    study = optuna.create_study(study_name = env_name + lr_schedule, direction = "maximize", sampler=TPESampler(seed=seed_val))
    study.optimize(obj, n_trials = trials)

    # Saving the study
    with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/" + lr_schedule + "_lr_" + env_name + ".pkl", "wb") as fout:
        pickle.dump(study, fout)

# Cartpole Experiment Parameters
cartpole_timesteps = 400000
cartpole_min_lr = 0.00001 # 1 order of magnitude less than sb3 default
cartpole_max_lr = 0.01 # 2 orders of magnitude greater than sb3 default
cartpole_n_trials = 15
cartpole_rl_t = .1 # reward loss threshold for adaptive_t learning rate

cartpole_envs = ["CustomCartPole-v0", "CustomCartPole-v1", "CustomCartPole-v2", "CustomCartPole-v3"]
# loop for cartpole experiments, trains new model for each env and lr strategy
for env in cartpole_envs:
    for strategy in  lr_strategies:
        train_model(env, cartpole_timesteps, DQN, strategy, cartpole_min_lr, cartpole_max_lr, cartpole_n_trials, cartpole_rl_t)

gym.envs.register(
    id='CustomInvertedDoublePendulum-v0',
    entry_point='custom_envs.custom_inverted_double_pendulum:CustomInvertedDoublePendulumEnvV0',
)
gym.envs.register(
    id='CustomInvertedDoublePendulum-v1',
    entry_point='custom_envs.custom_inverted_double_pendulum:CustomInvertedDoublePendulumEnvV1',
)
gym.envs.register(
    id='CustomInvertedDoublePendulum-v2',
    entry_point='custom_envs.custom_inverted_double_pendulum:CustomInvertedDoublePendulumEnvV2',
)

# IDP Experiment Parameters
idp_timesteps = 1000000
idp_min_lr = 0.00003 # 1 order of magnitude less than sb3 default
idp_max_lr = 0.03 # 2 orders of magnitude greater than sb3 default
idp_n_trials = 15
idp_rl_t = 1 # reward loss threshold for adaptive_t learning rate


idp_envs = ["InvertedDoublePendulum-v4", "CustomInvertedDoublePendulum-v0", "CustomInvertedDoublePendulum-v1", "CustomInvertedDoublePendulum-v2"]
# loop for Inverted Double Pendulum experiments, trains new model for each env and lr strategy
for env in idp_envs:
    for strategy in lr_strategies:
        train_model(env, idp_timesteps, PPO, strategy, idp_min_lr, idp_max_lr, idp_n_trials, idp_rl_t)

# Bipedal Walker Experiment Parameters
bp_timesteps = 3000000
bp_min_lr = 0.00003 # 1 order of magnitude less than sb3 default
bp_max_lr = 0.03 # 2 orders of magnitude greater than sb3 default
bp_n_trials = 5

lr_strategies = ["constant", "linear", "exponential", "adaptive"]
bp_envs = ["BipedalWalker-v3"]

# loop for Bipedal Walker experiments, trains new model for each env and lr strategy
#for env in bp_envs:
#    for strategy in lr_strategies:
#        train_model(env, bp_timesteps, PPO, strategy, bp_min_lr, bp_max_lr, bp_n_trials)