import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import os
import pickle as pkl

import sys
#sys.path.append("/home/nl6/FYP/FYP-2024/")
sys.path.append("C:/Users/35385/Desktop/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV0',
)

gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV1',
    )

def objective_constant(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)

    env = make_vec_env("CustomCartPole-v0", n_envs=4)
    model = DQN("MlpPolicy", env, learning_rate=learning_rate, verbose=0)

    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join('best'),
                                log_path="logdir", eval_freq=5000, n_eval_episodes=10,
                                deterministic=True, render=False, callback_after_eval=reward_threshold_callback)

    model.learn(total_timesteps=10000, progress_bar=True, callback=eval_callback)
        
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    return mean_reward

# maximizes reward
study = optuna.create_study(study_name="TEST", direction='maximize', load_if_exists=True)
# Optimize the learning rate
study.optimize(objective_constant, n_trials=2)

# SAVING STUDIES
with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/Test_study.pkl", "wb") as fout:
    pkl.dump(study, fout)

load_study = pkl.load(open("C:/Users/natha/Desktop/FYP-2024/saved_studies/Test_study.pkl", "rb"))

print('Number of finished trials:', len(load_study.trials))
print('Best trial:', load_study.best_trial.params)

#fig1 = plot_optimization_history(load_study)
#fig2 = plot_param_importances(study)
##fig3 = plot_parallel_coordinate(study)

#fig1.show()
#fig2.show()
#fig3.show()

'''import gymnasium as gym
env = gym.make("CustomCartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()'''