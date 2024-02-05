import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
import os
import numpy as np

base_dir = 'C:/Users/natha/Desktop/FYP-2024/trained_models/'
logdir = "logs"

seed = 12

model1_dir = "trained_IDP_PPO_model"
model2_dir = "trained_IDP_SAC_model"
model3_dir = "trained_IDP_td3_model"

TIMESTEPS = 1_000_000

env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")

def train_model(algorithm, env, model_dir, tb_log_name):

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'), 
                                log_path=logdir, eval_freq=5000, n_eval_episodes=5, 
                                deterministic=True, render=False, callback_after_eval=stop_train_callback)

    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
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
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            if terminated or truncated:
                avg_reward.append(total_reward)
        results.append(np.mean(avg_reward))  
    avg_results = np.mean(results) 
    return avg_results

#train_model(PPO, env, model1_dir, "IDP_PPO")
#train_model(SAC, env, model2_dir, "IDP_SAC")
#train_model(TD3, env, model3_dir, "IDP_td3")

ppo_model = PPO.load(os.path.join(base_dir, model1_dir + '_best', 'best_model'))
sac_model = SAC.load(os.path.join(base_dir, model2_dir + '_best', 'best_model'))
td3_model = TD3.load(os.path.join(base_dir, model3_dir + '_best','best_model'))

avg_reward_ppo = evaluate_model(ppo_model, env)
avg_reward_sac = evaluate_model(sac_model, env)
avg_reward_td3 = evaluate_model(td3_model, env)

print("Average rewards (PPO):", avg_reward_ppo)
print("Average rewards (SAC):", avg_reward_sac)
print("Average rewards (TD3):", avg_reward_td3)