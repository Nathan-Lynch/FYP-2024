import gymnasium as gym
import optuna
import os
import pickle
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from utils import create_objective

import sys
#sys.path.append("/home/nl6/FYP/FYP-2024/")
sys.path.append("/home/nl6/FYP/FYP-2024")
from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1

#--------------------------------------------------#
'''FIX CODE: CURRENT CODE JUST TO PRODUCE RESULTS'''
#--------------------------------------------------#

# For each want environment, want constant, linear, exponential, adaptive learning rates

trained_dir = "/home/nl6/FYP/FYP-2024/trained_models"
logdir = '/home/nl6/FYP/FYP-2024/logs/cartpole/'

model_constant_dir = "cartpole_constant_lr"
model_linear_dir = "cartpole_linear_lr"
model_exponential_dir = "cartpole_exponential_lr"
model_adaptive_dir = "cartpole_adaptive_lr"

# Default cartpole

env = gym.make("CartPole-v1", render_mode = None)
vec_env = make_vec_env("CartPole-v1", n_envs = 1)

cartpole_timesteps = 250000
reward_cb = StopTrainingOnRewardThreshold(reward_threshold = 500, verbose = 1) # stays the same for all cartpole environments

# constant lr
# maybe vec_env???
'''eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_constant_dir + '_best'),
                        log_path=logdir, eval_freq=10000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "constant", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/constant_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# linear decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_linear_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "linear", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/linear_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# exponential decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_exponential_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "exponential", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/exponential_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# Adaptive lr: no sense  for default cartpole: 
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_adaptive_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callback_list = [eval_cb]

obj = create_objective("CartPole-v1", DQN, cartpole_timesteps, logdir, callback_list, "adaptive", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/adaptive_lr_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# CustomCartpole-v0

trained_dir = "/home/nl6/FYP/FYP-2024/trained_models"
logdir = '/home/nl6/FYP/FYP-2024/logs/custom_cartpole_v0/'

model_constant_dir = "custom_cartpole-v0_constant_lr"
model_linear_dir = "custom_cartpole-v0_linear_lr"
model_exponential_dir = "custom_cartpole-v0_exponential_lr"
model_adaptive_dir = "custom_cartpole-v0_adaptive_lr"

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV0',
    )

env = gym.make("CustomCartPole-v0", render_mode = None)
vec_env = make_vec_env("CustomCartPole-v0", n_envs = 1)

# constant lr
# maybe vec_env???
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_constant_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v0", DQN, cartpole_timesteps, logdir, callbacks, "constant", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/constant_lr_custom_cartpole-v0.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# linear decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_linear_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v0", DQN, cartpole_timesteps, logdir, callbacks, "linear", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/linear_lr_custom_cartpole-v0.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# exponential decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_exponential_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v0", DQN, cartpole_timesteps, logdir, callbacks, "exponential", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/exponential_lr_custom_cartpole-v0.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# Adaptive lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_adaptive_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callback_list = [eval_cb]

obj = create_objective("CustomCartPole-v0", DQN, cartpole_timesteps, logdir, callback_list, "adaptive", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/adaptive_lr_custom_cartpole-v0.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)'''

# CustomCartpole-v1

trained_dir = "/home/nl6/FYP/FYP-2024/trained_models"
logdir = '/home/nl6/FYP/FYP-2024/logs/custom_cartpole_v1/'

model_constant_dir = "custom_cartpole-v1_constant_lr"
model_linear_dir = "custom_cartpole-v1_linear_lr"
model_exponential_dir = "custom_cartpole-v1_exponential_lr"
model_adaptive_dir = "custom_cartpole-v1_adaptive_lr"

gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV1',
    )

env = gym.make("CustomCartPole-v1", render_mode = None)
vec_env = make_vec_env("CustomCartPole-v1", n_envs = 1)

# constant lr
# maybe vec_env???
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_constant_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "constant", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/constant_lr_custom_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# linear decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_linear_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "linear", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/linear_lr_custom_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# exponential decreasing lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_exponential_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callbacks = CallbackList([eval_cb])

obj = create_objective("CustomCartPole-v1", DQN, cartpole_timesteps, logdir, callbacks, "exponential", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/exponential_lr_custom_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)

# Adaptive lr
eval_cb = EvalCallback(vec_env, best_model_save_path=os.path.join(trained_dir, model_adaptive_dir + '_best'),
                        log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                        deterministic=True, render=False)

callback_list = [eval_cb]

obj = create_objective("CustomCartPole-v1", DQN, cartpole_timesteps, logdir, callback_list, "adaptive", 0.0001, 0.01)

cartpole_study = optuna.create_study(direction = "maximize")
cartpole_study.optimize(obj, n_trials = 10)

with open("/home/nl6/FYP/FYP-2024/saved_studies/adaptive_lr_custom_cartpole-v1.pkl", "wb") as fout:
    pickle.dump(cartpole_study, fout)