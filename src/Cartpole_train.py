import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
import os
import numpy as np

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models/'
logdir = "logs"

model_dir = "trained_cartpole_dqn_model"

TIMESTEPS = 1_000_000

env = gym.make("CartPole-v1", render_mode="rgb_array")

def train_model(algorithm, env, model_dir, tb_log_name):

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, min_evals=10, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'), 
                                log_path=logdir, eval_freq=25_000, n_eval_episodes=10, 
                                deterministic=True, render=False, callback_after_eval=stop_train_callback)

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir, learning_rate = 0.0001)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=tb_log_name, callback=eval_callback)
    return model

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
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            if terminated or truncated:
                avg_reward.append(total_reward)
        results.append(np.mean(avg_reward))
    avg_results = np.mean(results) 
    return avg_results

train_model(DQN, env, model_dir, "cartpole_DQN")

dqn_model = DQN.load(os.path.join(base_dir, model_dir + '_best', 'best_model'))
avg_reward = evaluate_model(dqn_model, env)

print("Average rewards (DQN):", avg_reward)