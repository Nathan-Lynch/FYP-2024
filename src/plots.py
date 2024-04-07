import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def read_rewards(dir):
    '''
    Extracts the timesteps and corresponding values for a single tensorboard log file.

    :param dir: Directory of tensorboard logs.
    :return: Dictionary of {timestep: value} pairs.
    '''
    acc = event_accumulator.EventAccumulator(dir)
    acc.Reload()
    stats = acc.scalars.Items("eval/mean_reward")
    reward_values = {x.step: x.value for x in stats}
    return reward_values

def collect_rewards(subdirs, strategies):
    '''
    Collects rewards per timestep for each environment's experiments.

    :param subdirs: List of directories within logs directory.
    :param strategies: List of learning rate strategies.
    :return: Dictionary containing each environment's rewards per timestep for each learning rate strategy.
    '''
    env_rewards = {}

    for dir in subdirs:
        env_name = os.path.basename(dir)
        env_rewards[env_name] = {}

        for strategy in strategies:
            strat_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d.startswith(strategy)]
            rewards_collected = {}

            for strat_dir in strat_dirs:
                logdir = os.path.join(dir, strat_dir)
                rewards = read_rewards(logdir)

                for step, reward in rewards.items():
                    step = round(step / 10000) * 10000
                    if step not in rewards_collected:
                        rewards_collected[step] = []
                    rewards_collected[step].append(reward)

            env_rewards[env_name][strategy] = rewards_collected

    return env_rewards

def plot_experiment_reward_data_with_variance(env_name, env_data):
    '''
    Plots the timestep (x), average reward (y) pairs, and standard deviation bands for an experiment.

    :param env_name: Name of the environment.
    :param env_data: Dictionary containing each strategy and their collected rewards for env_name.
    '''
    plt.figure(figsize=(10, 6))

    for strategy, data in env_data.items():
        steps = sorted(data.keys())
        means = []
        std_devs = []
        for step in steps:
            rewards = data[step]
            means.append(np.mean(rewards))
            std_devs.append(np.std(rewards))

        steps = np.array(steps)
        means = np.array(means)
        std_devs = np.array(std_devs)
        
        plt.plot(steps, means, label=strategy)
        plt.fill_between(steps, means - std_devs, means + std_devs, alpha=0.15)

    filename = f"{env_name}_average_training_rewards_with_variance.png"
    plt.title(f"Average Rewards and Variance for {env_name}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend(loc="lower right")
    plt.grid(visible=True, axis="both", color="grey", linewidth=1, alpha=0.5)
    plt.savefig(f"C:/Users/natha/Desktop/FYP-2024/plots/{filename}")
    plt.close()  

logdir = 'C:/Users/natha/Desktop/FYP-2024/logs/'
subdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]

lr_strategies = ["constant", "linear", "exponential", "adaptive"]

data = collect_rewards(subdirs, lr_strategies)

for env_name, env_data in data.items():
    plot_experiment_reward_data_with_variance(env_name, env_data)