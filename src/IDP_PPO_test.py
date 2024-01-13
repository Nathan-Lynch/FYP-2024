import gymnasium as gym
import torch
from PPO import PPO
from plots import plot_rewards_test
import matplotlib.pyplot as plt

plt.figure(2)
rewards = []

def test_IDP_PPO(episodes):
    total_reward = 0

    env = gym.make("InvertedDoublePendulum-v4", render_mode = 'human')
    
    model = PPO()
    model.policy_module.load_state_dict(torch.load('C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_IDP_PPO_model.pth'))
    model.policy_module.eval()
    
    for episode in range(episodes):
        total_reward = 0
        observation, _ = env.reset()

        truncated = False
        terminated = False

        while not truncated or not terminated:
            state_tensor = {'observation': torch.tensor(observation, dtype=torch.float32)}
            with torch.no_grad():
                action_params = model.policy_module(state_tensor)

            action_mean = action_params['loc'].cpu().numpy()
            action = torch.tanh(action_mean)

            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                rewards.append(total_reward)
                break            
        
        print(f"Episode {episode}: Total Reward: {total_reward}")

    plot_rewards_test(rewards)

    env.close()

test_IDP_PPO(100)
