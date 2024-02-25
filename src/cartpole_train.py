import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import RewardLossCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable, Optional
from evaluate_model import evaluate_model
import os
import numpy as np

#base_dir = '/home/nl6/FYP/FYP-2024/trained_models'
base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models'
model_dir = "trained_cartpole_dqn_model"
model2_dir = "trained_cartpole_dqn_linear_lr_model"
logdir = 'C:/Users/natha/Desktop/FYP-2024/logs/cartpole/'
#logdir = "/home/nl6/FYP/FYP-2024/logs"

TIMESTEPS = 5_000_000

env = gym.make("CartPole-v1", render_mode=None)
vec_env = make_vec_env("CartPole-v1", n_envs=16)

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
        lr = linear_schedule(0.001)

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'),
                                 log_path=logdir, eval_freq=5000, n_eval_episodes=10,
                                 deterministic=True, render=False, callback_after_eval=reward_threshold_callback)
    
    rl = RewardLossCallback()

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir, learning_rate=lr)

    callback = CallbackList([eval_callback, rl])

    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

const_lr_rewards = []
decreasing_lr_rewards = []
for i in range(5):
    train_model(DQN, vec_env, model_dir, "cartpole_dqn", schedule=False) # trains in roughly 2.36 million timesteps
    train_model(DQN, vec_env, model2_dir, "cartpole_dqn_linear_lr", schedule=True) # trains in roughky 1.792 million timesteps

    dqn_model1 = DQN.load(os.path.join(base_dir, model_dir + '_best', 'best_model'))
    dqn_model2 = DQN.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))

    const_lr_rewards.append(evaluate_model(dqn_model1, env))
    decreasing_lr_rewards.append(evaluate_model(dqn_model2, env))

print(np.mean(const_lr_rewards))
print(np.mean(decreasing_lr_rewards))

train_model(DQN, vec_env, "test", "cartpole_dqn_rl_test", schedule=False)