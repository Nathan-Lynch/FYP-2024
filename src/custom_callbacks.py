import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardLossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLossCallback, self).__init__(verbose)
        self.prev_reward = None
        self.reward_loss = None

    def _on_step(self):
        current_reward = self.locals["rewards"][0]
        if self.prev_reward is not None:
            reward_loss = current_reward - self.prev_reward
            self.reward_loss = reward_loss
        self.prev_reward = current_reward

        if self.reward_loss is not None:
            if self.reward_loss >= 0:
                self.model.learning_rate = self.model.learning_rate * 1.1
            else:
                self.model.learning_rate = self.model.learning_rate * 0.9

        return True


