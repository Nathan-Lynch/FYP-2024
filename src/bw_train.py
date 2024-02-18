import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import RewardLossCallback
from evaluate_model import evaluate_model
import os
import numpy as np

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models'
logdir = "logs"

model1_dir = "trained_bw_ppo_model"
model2_dir = "trained_bw_sac_model"
model3_dir = "trained_bw_td3_model"

TIMESTEPS = 5_000_000

env = gym.make("BipedalWalker-v3", render_mode=None)

def train_model(algorithm, env, model_dir, tb_log_name):

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'),
                                 log_path=logdir, eval_freq=10000, n_eval_episodes=10,
                                 deterministic=True, render=False, callback_after_eval=reward_threshold_callback)

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)

    callback = CallbackList([eval_callback])

    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

train_model(PPO, env, model1_dir, "bw_ppo")
train_model(SAC, env, model2_dir, "bw_sac")
train_model(TD3, env, model3_dir, "bw_td3")

ppo_model = PPO.load(os.path.join(base_dir, model1_dir + '_best', 'best_model'))
sac_model = SAC.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))
td3_model = TD3.load(os.path.join(base_dir, model3_dir + '_best', 'best_model'))

avg_reward_ppo = evaluate_model(ppo_model, env)
avg_reward_sac = evaluate_model(sac_model, env)
avg_reward_td3 = evaluate_model(td3_model, env)

print("Average rewards (PPO):", avg_reward_ppo)
print("Average rewards (SAC):", avg_reward_sac)
print("Average rewards (TD3):", avg_reward_td3)