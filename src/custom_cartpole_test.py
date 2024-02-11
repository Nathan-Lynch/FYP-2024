import gymnasium as gym
from stable_baselines3 import DQN

import sys
sys.path.append("C:/Users/natha/Desktop/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnv
import numpy as np
import os

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnv',
    )

env = gym.make("CustomCartPole-v0", render_mode = None)

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models/'
logdir = "logs"

model1_dir = 'trained_custom_cartpole_dqn_model'
#model2_dir = "trained_idp_sac_model"

def evaluate_model(model, env):
    results = []
    for _ in range(10):
        avg_reward = []
        for i in range(5):
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

model1 = DQN.load(os.path.join(base_dir, model1_dir + '_best', 'best_model'))
#model2 = DQN.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))

avg_reward_cartpole = evaluate_model(model1, env)
#avg_reward = evaluate_model(model2, env)

print("Average rewards (PPO):", avg_reward_cartpole)
#print("Average rewards (SAC):", avg_reward_sac)
