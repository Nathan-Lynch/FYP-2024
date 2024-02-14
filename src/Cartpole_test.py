import gymnasium as gym
import torch
from Models.DQN import DQN
from plots import plot_rewards_test
import matplotlib.pyplot as plt

plt.figure(2)
rewards = []

def test_cartpole(episodes):
    total_reward = 0

    env = gym.make("CartPole-v1", render_mode = None)
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load('C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_cartpole_model.pth'))
    model.eval()
    
    for episode in range(episodes):
        total_reward = 0
        observation, _ = env.reset()

        truncated = False
        terminated = False

        while not truncated or not terminated:
            state_tensor = torch.tensor(observation, dtype=torch.float32)
            action = model(state_tensor).max(0)[1].item()
            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                rewards.append(total_reward)
                break            
        
        print(f"Episode {episode}: Total Reward: {total_reward}")

    plot_rewards_test(rewards)

    env.close()

test_cartpole(100)
