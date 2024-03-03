from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from typing import Callable
import numpy as np
import math
import os

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    # From:
    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress

    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(final_value, progress_remaining * initial_value)

    return func

def exponential_schedule(initial_value: float, decay_factor: float, final_value: float) -> Callable[[float], float]:

    """
    Exponential learning rate schedule.

    param: initial_value: Initial learning rate.
    param: decay_factor: Rate of decay
    return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        param: progress_remaining:
        return: current learning rate
        """
        return max(final_value, initial_value * math.exp(-(1-progress_remaining) * decay_factor))

    return func

class AdaptiveLearningRateScheduler:
    def __init__(self, initial_lr=0.001, increase_factor=1.0001, decrease_factor=0.9999):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.last_reward = None

    def adjust_learning_rate(self, reward):
        if self.last_reward is not None:
            reward_diff = reward - self.last_reward
            if reward_diff >= 0:
                self.current_lr *= self.increase_factor
            else:
                self.current_lr *= self.decrease_factor
            # LOOK INTO FOR MIN/MAX learning rates
            # self.current_lr = max(min_lr, min(self.current_lr, max_lr))
        self.last_reward = reward

    def get_current_lr(self):
        return self.current_lr

def create_objective(env_name, model_name, timesteps, logdir, callbacks, lr_schedule, min_lr, max_lr):
    '''
    Creates a custom objective function for optimization based on the environment, model, and learning rate schedule

    param env:
    param model:
    #TODO 
    '''
    def objective(trial):
        env = make_vec_env(env_name, n_envs = 16)
        if lr_schedule == "constant":
            learning_rate = trial.suggest_float('learning_rate', min_lr, max_lr, log=True)

            model = model_name("MlpPolicy", env, learning_rate=learning_rate, verbose=0, tensorboard_log = logdir)
            model.learn(total_timesteps=timesteps, progress_bar=True, callback=callbacks, tb_log_name = "constant_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
            return mean_reward

        elif lr_schedule == "linear":
            initial_lr = trial.suggest_float('initial_lr', min_lr, max_lr, log=True)
            final_lr = trial.suggest_float('final_lr', min_lr/10 , min_lr, log=True)

            final_lr = min(final_lr, initial_lr)

            schedule = linear_schedule(initial_lr, final_lr)

            model = model_name("MlpPolicy", env, learning_rate=schedule, verbose=0, tensorboard_log = logdir)
            model.learn(total_timesteps=timesteps, progress_bar=True, callback=callbacks, tb_log_name = "linear_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
            return mean_reward
        
    return objective


# LOOK INTO: MAY NOT NEED
def evaluate_model(model, env):
    results = []
    for _ in range(10):
        avg_reward = []
        for _ in range(5):
            total_reward = 0
            observation, info = env.reset()

            truncated = False
            terminated = False

            while not truncated and not terminated:
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            if terminated or truncated:
                avg_reward.append(total_reward)
        results.append(np.mean(avg_reward))  
    avg_results = np.mean(results) 
    return avg_results