import gymnasium as gym
from plots import plot_rewards_test
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

plt.figure(2)
rewards = []

def test_IDP_PPO(episodes):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    model = PPO.load("C:/Users/natha/Desktop/FYP-2024/trained_models/trained_BW_PPO_model")

    for episode in range(episodes):
        total_reward = 0
        observation, _ = env.reset()

        truncated = False
        terminated = False

        while not truncated and not terminated:
            action, _state = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                rewards.append(total_reward)
                break

        print(f"Episode {episode+1}: Total Reward: {total_reward}")

    plot_rewards_test(rewards, 'BW-v3_Test')

    env.close()

test_IDP_PPO(100)