from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from custom_callbacks import AdaptiveLRCallback
from typing import Callable
import gymnasium as gym
import numpy as np
import math
import os

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    # From:
    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    """
    Schedule for Linearly decreasing learning rate.

    :param initial_value: Initial learning rate
    :param final_value: Minimum value for the learning rate
    :return: schedule that computes current learning rate depending on remaining progress

    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: Timesteps remaining during training
        :return: Updated learning rate
        """
        return max(final_value, progress_remaining * initial_value)

    return func

def exponential_schedule(initial_value: float, final_value: float, decay_factor: float) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.

    :param: initial_value: Initial learning rate
    :param: final_value: Minimum value for the learning rate
    :param: decay_factor: Rate of decay
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: Timesteps remaining during training
        :return: Updated learning rate
        """
        return max(final_value, initial_value * math.exp(-(1-progress_remaining) * decay_factor))

    return func

class AdaptiveLearningRate:
    '''
    DESCRIPTION TODO

    :param initial_lr: Initial learning rate
    :param top_lr: Maximum value allowed for learning rate
    :param bottom_lr: Minimum value allowed for learning rate
    :param increase_factor: Value to increase learning rate
    :param decrease_factor: Value to decrease learning rate
    '''

    def __init__(self, initial_lr, top_lr, bottom_lr, increase_factor, decrease_factor):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.top_lr = top_lr
        self.bottom_lr = bottom_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.episode_rewards = [] # Tracks current episode rewards

    def adjust_learning_rate(self, reward):
        '''
        Calculates the reward mean and adjusts the learning rate accordingly.
        If reward loss is positive increase learning rate by increase factor.
        If reward loss is negative decrease learning rate by decrease factor.

        :param reward: Reward obtained from latest timestep
        '''
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > 0:
            reward_loss = reward - np.mean(self.episode_rewards)
            if reward_loss >= 0:
                self.current_lr = min(self.current_lr * self.increase_factor, self.top_lr)
            else:
                self.current_lr = max(self.current_lr * self.decrease_factor, self.bottom_lr)

    def get_current_lr(self):
        '''
        :return: Current learning rate
        '''
        return self.current_lr
    
    def reset_episode_rewards(self):
        '''
        Resets self.current_rewards when called
        '''
        self.episode_rewards = []

def create_objective(env_name, model_name, timesteps, logdir, callback, lr_schedule, min_lr, max_lr):
    '''
    Creates a custom objective function for optimization based on the environment, 
    model, and learning rate stategy

    :param env_name: Name of the environment
    :param model_name: The model used for training
    :param timesteps: Number of timesteps for training
    :param logdir: Directory of the logged files
    :param callback: Callbacks to be used within training
    :param lr_schedule: Learning rate strategy to be used
    :param min_lr: Minimum value allowed for learning rate
    :param max_lr: Maximum value allowed for learning rate
    :return: objective
    '''
    def objective(trial):
        '''
        Maximizes the mean reward during training

        :param trial: Trial of Optuna study TODO make clearer
        '''
        env = gym.make(env_name, render_mode = None)
        if lr_schedule == "constant":
            learning_rate = trial.suggest_float('learning_rate', min_lr, max_lr, log = True)

            model = model_name("MlpPolicy", env, learning_rate = learning_rate, verbose = 0, tensorboard_log = logdir)
            model.learn(total_timesteps = timesteps, progress_bar = True, callback = callback, tb_log_name = "constant_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes = 10)[0]
            return mean_reward

        elif lr_schedule == "linear":
            initial_lr = trial.suggest_float('initial_lr', min_lr, max_lr, log = True)
            final_lr = trial.suggest_float('final_lr', min_lr/10 , min_lr, log = True)

            final_lr = min(final_lr, initial_lr)

            schedule = linear_schedule(initial_lr, final_lr)

            model = model_name("MlpPolicy", env, learning_rate = schedule, verbose = 0, tensorboard_log = logdir)
            model.learn(total_timesteps = timesteps, progress_bar = True, callback = callback, tb_log_name = "linear_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes = 10)[0]
            return mean_reward

        elif lr_schedule == "exponential":
            initial_lr = trial.suggest_float('initial_lr', min_lr, max_lr, log = True)
            final_lr = trial.suggest_float('final_lr', min_lr/10 , min_lr, log = True)
            decay_rate = trial.suggest_float('decay_rate', 0.01, 0.99)

            final_lr = min(final_lr, initial_lr)

            schedule = exponential_schedule(initial_lr, final_lr, decay_rate)

            model = model_name("MlpPolicy", env, learning_rate = schedule, verbose = 0, tensorboard_log = logdir)
            model.learn(total_timesteps = timesteps, progress_bar = True, callback = callback, tb_log_name = "exponential_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes = 10)[0]
            return mean_reward

        else:
            initial_lr = trial.suggest_float('initial_lr', min_lr, max_lr, log = True)
            top_lr = trial.suggest_float('top_lr', initial_lr*2, initial_lr*10, log = True)
            bottom_lr = trial.suggest_float('bottom_lr', initial_lr/10, initial_lr/2, log = True)
            adjustment_factor = trial.suggest_float('adjustment_factor', 0.01, 0.1, log = True)

            # TODO FIX NAMING OF THESE
            schedule = AdaptiveLearningRate(initial_lr = initial_lr, top_lr = top_lr, bottom_lr = bottom_lr, increase_factor = 1+adjustment_factor, decrease_factor = 1-adjustment_factor)
            scheduler = AdaptiveLRCallback(schedule)

            callback.append(scheduler)
            callbacks = CallbackList(callback)

            model = model_name("MlpPolicy", env, learning_rate = schedule.get_current_lr(), verbose = 0, tensorboard_log = logdir)
            model.learn(total_timesteps = timesteps, progress_bar = True, callback = callbacks, tb_log_name = "adaptive_lr")

            mean_reward = evaluate_policy(model, env, n_eval_episodes = 10)[0]
            return mean_reward

    return objective