import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

class RewardLossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLossCallback, self).__init__(verbose)
        self.prev_episode_reward = None
        self.reward_loss = None

    def _on_step(self):
        current_reward = self.locals["rewards"][0]
        if self.prev_reward is not None:
            reward_loss = current_reward - self.prev_reward
            self.reward_loss = reward_loss
        self.prev_reward = current_reward

        #reward_losses.append(reward_loss)

        return True
    
class AdaptiveLRCallback(BaseCallback):
    def __init__(self, lr_scheduler, verbose=0):
        super(AdaptiveLRCallback, self).__init__(verbose)
        self.lr_scheduler = lr_scheduler

    def _on_step(self):
        reward = self.locals['rewards'][-1]
        self.lr_scheduler.adjust_learning_rate(reward)
        new_lr = self.lr_scheduler.get_current_lr()
        self.model.policy.optimizer.param_groups[0]['lr'] = new_lr

        self.logger.record(key="custom/learning_rate", value=new_lr)


        return True


