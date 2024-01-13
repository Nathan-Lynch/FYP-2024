import gymnasium as gym
import torch
from DQN import DQN

env = gym.make("InvertedDoublePendulum-v4", render_mode = "human")

n_actions = env.action_space.shape[0]
state, info = env.reset()
n_observations = len(state)

model = DQN(n_observations, n_actions)
model.load_state_dict(torch.load('C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_IDP_DQN_model.pth'))
model.eval()

total_reward = 0

def test_cartpole(model, episodes):    
    for episode in range(1, episodes+1):
        total_reward = 0
        observation, info = env.reset()

        truncated = False
        terminated = False

        while not truncated or not terminated:
            state_tensor = torch.tensor(observation, dtype=torch.float32)
            action = model(state_tensor).max(0)[1].item()
            observation, reward, terminated, truncated, info = env.step([action])

            total_reward += reward

            if total_reward >= 500:
                break            
        
        print(f"Episode {episode}: Total Reward: {total_reward}")

    env.close()

test_cartpole(model, 2)

