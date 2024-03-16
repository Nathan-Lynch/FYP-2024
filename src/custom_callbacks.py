from stable_baselines3.common.callbacks import BaseCallback

class AdaptiveLRCallback(BaseCallback):
    '''
    Custom Callback that extends gymnasium BaseCallback.
    Allows learning rate to be changed based off real time metrics (reward loss)
    every timestep.

    :param lr_scheduler: AdaptiveLearningRate object.
    :param verbose: TODO figure out if necessary.
    '''
    def __init__(self, lr_scheduler, verbose=0):
        super(AdaptiveLRCallback, self).__init__(verbose)
        self.lr_scheduler = lr_scheduler

    def _on_step(self):
        '''
        Updates the learning rate every timestep and logs the values to tensorboard.
        '''
        # Resets episode rewards if episode terminates.
        if self.locals["dones"][-1]:
            self.lr_scheduler.reset_episode_rewards()

        reward = self.locals['rewards'][-1]
        self.lr_scheduler.adjust_learning_rate(reward)
        new_lr = self.lr_scheduler.get_current_lr()
        self.model.policy.optimizer.param_groups[0]['lr'] = new_lr

        self.logger.record(key="train/learning_rate", value=new_lr)

        return True