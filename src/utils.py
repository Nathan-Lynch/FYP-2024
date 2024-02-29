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


def evaluate_model(model, env):
    results = []
    for _ in range(10):
        avg_reward = []
        for _ in range(5):
            total_reward = 0
            observation, _ = env.reset()

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