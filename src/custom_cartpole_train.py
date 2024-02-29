import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import os
import numpy as np
from custom_callbacks import RewardLossCallback
from evaluate_model import evaluate_model


import sys
sys.path.append("/home/nl6/FYP/FYP-2024/")
#sys.path.append("C:/Users/35385/Desktop/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnv

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnv',
    )

base_dir = '/home/nl6/FYP/FYP-2024/trained_models'
#base_dir = 'C:/Users/35385/Desktop/FYP-2024/trained_models'
model1_dir = "trained_custom_cartpole_dqn_model"
model2_dir = "trained_custom_cartpole_dqn_linear_lr_model"
#logdir = 'C:/Users/35385/Desktop/FYP-2024/logs/cartpole'
logdir = "/home/nl6/FYP/FYP-2024/logs/custom_cartpole/"

TIMESTEPS = 5_000_000

env = gym.make("CustomCartPole-v0", render_mode=None)
vec_env = make_vec_env("CustomCartPole-v0", n_envs=8)

def linear_schedule(initial_value: float) -> Callable[[float], float]:

    # From:
    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train_model(algorithm, env, model_dir, tb_log_name, schedule):

    if schedule == False:
        lr = 0.001
    else:
        lr = linear_schedule(0.01)

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'),
                                 log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                                 deterministic=True, render=False, callback_after_eval=reward_threshold_callback)
    
    rl = RewardLossCallback()

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir, learning_rate=lr)

    callback = CallbackList([eval_callback])

    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

const_lr_rewards = []
decreasing_lr_rewards = []
for i in range(5):
    train_model(DQN, env, model1_dir, "custom_cartpole_dqn", schedule=False)
    train_model(DQN, vec_env, model2_dir, "custom_cartpole_dqn_linear_lr", schedule=True)

    dqn_model1 = DQN.load(os.path.join(base_dir, model1_dir + '_best', 'best_model'))
    dqn_model2 = DQN.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))

    const_lr_rewards.append(evaluate_model(dqn_model1, env))
    decreasing_lr_rewards.append(evaluate_model(dqn_model2, env))

print(np.mean(const_lr_rewards))
print(np.mean(decreasing_lr_rewards))
