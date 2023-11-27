import gymnasium as gym
import torch
from DQN import DQN

def test_cartpole(episodes):
    total_reward = 0

    env = gym.make("CartPole-v1", render_mode = "human")
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load('trained_models/trained_cartpole_model.pth'))
    model.eval()
    
    for episode in range(1, episodes+1):
        total_reward = 0
        observation, _ = env.reset()

        truncated = False
        terminated = False

        while not truncated or not terminated:
            state_tensor = torch.tensor(observation, dtype=torch.float32)
            action = model(state_tensor).max(0)[1].item()
            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if total_reward == 500:
                break            
        
        print(f"Episode {episode}: Total Reward: {total_reward}")

    env.close()

test_cartpole(2)
