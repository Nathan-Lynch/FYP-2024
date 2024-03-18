import optuna
from optuna.visualization import plot_contour, plot_edf, plot_hypervolume_history, plot_intermediate_values, plot_optimization_history, plot_terminator_improvement, plot_parallel_coordinate
from optuna.visualization import plot_param_importances, plot_pareto_front, plot_rank, plot_slice, plot_terminator_improvement, plot_timeline, is_available
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import os
import pickle

import sys
#sys.path.append("/home/nl6/FYP/FYP-2024/")
sys.path.append("C:/Users/natha/Desktop/FYP-2024")

from custom_envs.custom_cartpole import CustomCartPoleEnvV0, CustomCartPoleEnvV1, CustomCartPoleEnvV2

gym.envs.register(
    id='CustomCartPole-v0',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV0',
)

gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV1',
    )

gym.envs.register(
    id='CustomCartPole-v2',
    entry_point='custom_envs.custom_cartpole:CustomCartPoleEnvV2',
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
#study = optuna.create_study(study_name="TEST", direction='maximize', load_if_exists=True)
# Optimize the learning rate
#study.optimize(objective_constant, n_trials=2)

# SAVING STUDIES
#with open("C:/Users/natha/Desktop/FYP-2024/saved_studies/Test_study.pkl", "wb") as fout:
#    pkl.dump(study, fout)

#load_study = pkl.load(open("C:/Users/natha/Desktop/FYP-2024/saved_studies/Test_study.pkl", "rb"))

#print('Number of finished trials:', len(load_study.trials))
#print('Best trial:', load_study.best_trial.params)

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

'''load_study = pickle.load(open("C:/Users/35385/Desktop/FYP-2024/saved_studies/adaptive_lr_custom_cartpole-v0.pkl", "rb"))
fig1 = plot_contour(load_study)
fig2 = plot_edf(load_study) # no, probably not important
#fig3 = plot_hypervolume_history(load_study)
fig4 = plot_intermediate_values(load_study) # no, nothing showed up
fig5 = plot_optimization_history(load_study) # no?, doesnt give enough info for 10 trials
fig6 = plot_parallel_coordinate(load_study) # potentially gives info to talk about
#fig7 = plot_pareto_front(load_study)
fig8 = plot_rank(load_study) # potentially
fig9 = plot_slice(load_study) # no, dont know what plot is showing me
#fig10 = plot_terminator_improvement(load_study)
fig11 = plot_timeline(load_study) # no, not important
fig12 = plot_param_importances(load_study) # maybe

fig1.show()
fig2.show()
#fig3.show()
fig4.show()
fig5.show()
fig6.show()
#fig7.show()
fig8.show()
fig9.show()
#fig10.show()
fig11.show()
fig12.show()'''



#print(load_study.best_trial.number)
#print(load_study.best_trial.params)

#load_study = pkl.load(open("C:/Users/natha/Desktop/FYP-2024/saved_studies/adaptive_lr_custom_cartpole-v1.pkl", "rb"))
#print(load_study.best_trial.number)
#print(load_study.best_trial.params)

gym.envs.register(
    id='CustomInvertedDoublePendulum-v0',
    entry_point='custom_envs.custom_inverted_double_cartpole:CustomInvertedDoublePendulumEnvV0',
)
gym.envs.register(
    id='CustomInvertedDoublePendulum-v1',
    entry_point='custom_envs.custom_inverted_double_cartpole:CustomInvertedDoublePendulumEnvV1',
)
gym.envs.register(
    id='CustomBipedalWalker-v0',
    entry_point='custom_envs.custom_bipedal_walker_env:CustomBipedalWalkerEnv',
)

import gymnasium as gym
env = gym.make("CustomInvertedDoublePendulum-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(100000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()