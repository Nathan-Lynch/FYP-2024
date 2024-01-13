import torch
import matplotlib.pyplot as plt

def plot_rewards_loss(axs, episode_rewards, episode_losses, show_result=False):
    rewards = torch.tensor(episode_rewards, dtype=torch.float)
    losses = torch.tensor(episode_losses, dtype=torch.float)

    axs[0].plot(rewards.numpy(), color='blue')
    axs[0].set_title('Reward per Episode')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Episode')

    axs[1].plot(losses.numpy(), color='red')
    axs[1].set_title('Average Loss per Episode')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Episode')

    plt.tight_layout()

    if len(rewards) >= 100:
        mean_reward = rewards.unfold(0, 100, 1).mean(1).view(-1)
        mean_reward = torch.cat((torch.zeros(99), mean_reward))
        axs[0].plot(mean_reward.numpy(), color='orange')

    plt.savefig('C:/Users/natha/Desktop/FYP-2024/src/plots/CartPole-v1.png')
    plt.pause(0.001)


def plot_rewards_test(episode_rewards):
    rewards = torch.tensor(episode_rewards, dtype=torch.float)

    plt.plot(rewards.numpy(), color='blue')
    plt.title('Reward per Episode') 
    plt.ylabel('Reward')  
    plt.xlabel('Episode') 

    plt.savefig('C:/Users/natha/Desktop/FYP-2024/src/plots/CartPole-v1_Test.png')