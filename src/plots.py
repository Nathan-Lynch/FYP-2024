import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def plot_experiemnt_reward_data(env_name, env_data):
    '''
    Plots the timestep (x), average reward (y) pairs for an experiment on a single figure.

    :param env_name: Name of the environment.
    :param env_data: Dictionary containing each strategy and their timestep, value pairs for env_name.
    '''
    plt.figure(figsize=(10, 6)) # Sets width, height in inches

    for strategy, data in env_data.items():
        steps = list(data.keys())
        rewards = list(data.values())
        plt.plot(steps, rewards, label=strategy)

    filename = f"{env_name}_average_training_rewards.png"
    plt.title(f"Average Rewards for {env_name}")
    plt.xlabel("steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(visible = True, axis = "both", color="grey", linewidth = 1, alpha = 0.5) # Adds grid lines on plot for clarity
    plt.savefig(f"C:/Users/35385/Desktop/FYP-2024/plots/{filename}")

def read_rewards(dir):
    '''
    Extracts the timesteps and corresponding values for a single tensorboard log file.

    :param dir: Directory of tensorboard logs
    :return: Dictionary of {timestep:value} pairs
    '''
    acc = event_accumulator.EventAccumulator(dir)
    acc.Reload()
    stats = acc.scalars.Items("eval/mean_reward")
    reward_values = {x.step:x.value for x in stats}
    return reward_values

def average_reward(subdirs, strategies):
    '''
    Calculates the average reward per timestep for each environments experiments.

    :param subdirs: List of directories within ~/logs TODO make clearer.
    :param strategies: List of learning rate strategies.
    :return: Dictionary containing each environments' average reward per timestep for each learning rate.
    '''
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

# Directories for extracting data
logdir = "C:/Users/35385/Desktop/FYP-2024/logs"
subdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]

# Learning rate strategies used within experiments
lr_strategies = ["constant", "linear", "exponential", "adaptive"]
data = average_reward(subdirs, lr_strategies)

# Plot and save figure for all experiments done
for env_name, env_data in data.items():
    plot_experiemnt_reward_data(env_name, env_data)