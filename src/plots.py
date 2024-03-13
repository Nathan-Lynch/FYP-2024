import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

logdir = "C:/Users/35385/Desktop/FYP-2024/logs"
subdir = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]

def plot_experiemnt_reward_data(env_name, env_data):
    plt.figure(figsize=(10, 6))

    for strategy, data in env_data.items():
        steps = list(data.keys())
        rewards = list(data.values())
        plt.plot(steps, rewards, label=strategy)

    filename = f"{env_name}_average_training_rewards.png"
    plt.title(f"Average Rewards for {env_name}")
    plt.xlabel("steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(visible = True, axis = "both", color="grey", linewidth = 1, alpha = 0.5)
    plt.savefig(f"C:/Users/35385/Desktop/FYP-2024/plots/{filename}")
    plt.show()

def read_rewards(dir):
    acc = event_accumulator.EventAccumulator(dir)
    acc.Reload()
    stats = acc.scalars.Items("eval/mean_reward")
    reward_values = {x.step:x.value for x in stats}
    return reward_values

def average_reward(subdirs, strategies):
    env_rewards = {}

    for dir in subdirs:
        env_name = os.path.basename(dir)
        env_rewards[env_name] = {}

        for strategy in strategies:
            strat_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d.startswith(strategy)]
            env_step_rewards = {}
            env_step_counts = {}

            for strat_dir in strat_dirs:
                logdir = os.path.join(dir, strat_dir)
                rewards = read_rewards(logdir)

                for step, reward in rewards.items():
                    if step not in env_step_rewards:
                        env_step_rewards[step] = 0
                        env_step_counts[step] = 0
                    env_step_rewards[step] += reward
                    env_step_counts[step] += 1
            average_rewards = {step: env_step_rewards[step] / env_step_counts[step] for step in env_step_rewards}
            env_rewards[env_name][strategy] = average_rewards

    return env_rewards

lr_strategies = ["constant", "linear", "exponential", "adaptive"]
data = average_reward(subdir, lr_strategies)

for env_name, strategy in data.items():
    plot_experiemnt_reward_data(env_name, strategy)