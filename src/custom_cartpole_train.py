from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
import gymnasium as gym
import os

import sys
sys.path.append("C:/Users/natha/Desktop/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnv

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnv',
    )

env = gym.make("CustomCartPole-v0", render_mode = "human")

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models/'
model_dir = "trained_custom_cartpole_dqn_model"
logdir = "logs"

TIMESTEPS = 1_000_000

env = gym.make("CustomCartPole-v0", render_mode=None)

def train_model(algorithm, env, model_dir, tb_log_name):

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, min_evals=10, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'), 
                                log_path=logdir, eval_freq=25_000, n_eval_episodes=10, 
                                deterministic=True, render=False, callback_after_eval=stop_train_callback)

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=eval_callback)
    return model

train_model(DQN, env, model_dir, "custom_cartpole_dqn")
