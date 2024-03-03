import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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


