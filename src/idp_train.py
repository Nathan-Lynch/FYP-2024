import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import RewardLossCallback
from typing import Callable, Optional
from evaluate_model import evaluate_model
import os
import numpy as np

base_dir = '/home/nl6/FYP/FYP-2024/trained_models'
model1_dir = "trained_idp_ppo_model"
model2_dir = "trained_idp_ppo_linear_lr_model"
logdir = "/home/nl6/FYP/FYP-2024/logs/idp"

TIMESTEPS = 5_000_000

env = gym.make("InvertedDoublePendulum-v4", render_mode=None)
vec_env = make_vec_env("InvertedDoublePendulum-v4", n_envs = 16)

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

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold = 9360, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'), 
                                log_path=logdir, eval_freq=25000, n_eval_episodes=25, 
                                deterministic=True, render=False, callback_after_eval=reward_threshold_callback)

    callback = CallbackList([eval_callback])

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir, learning_rate = lr)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

const_lr_rewards = []
decreasing_lr_rewards = []

for i in range(5):
    train_model(PPO, vec_env, model1_dir, "idp_ppo", schedule=False)
    train_model(PPO, vec_env, model2_dir, "idp_ppo_linear_lr", schedule=True)

    ppo_model1 = PPO.load(os.path.join(base_dir, model1_dir + '_best', 'best_model'))
    ppo_model2 = PPO.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))

    const_lr_rewards.append(evaluate_model(ppo_model1, env))
    decreasing_lr_rewards.append(evaluate_model(ppo_model2, env))

print(np.mean(const_lr_rewards))
print(np.mean(decreasing_lr_rewards))