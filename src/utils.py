from typing import Callable
import numpy as np

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

def adaptive_learning_rate_scheduler(initial_value: float) -> Callable[[float], float]:
    '''
    learning rate adjusted by reward loss

    :param learning_rate: Initial learning rate
    :return: schedule that computes new learning rate based on reward loss
    '''
    def func(reward_loss: float) -> float:
        if reward_loss >= 0:
            return initial_value * 1.0001
        else:
            return initial_value * 0.99999
        
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
            # Optionally, enforce minimum and maximum learning rates
            # self.current_lr = max(min_lr, min(self.current_lr, max_lr))
        self.last_reward = reward

    def get_current_lr(self):
        return self.current_lr


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