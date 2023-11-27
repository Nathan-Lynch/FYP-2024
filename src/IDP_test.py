import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("InvertedDoublePendulum-v4", render_mode = "human")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Get the number of actions from the gym action space
n_actions = env.action_space.shape[0]
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

model = DQN(n_observations, n_actions)

# Load the trained model state dictionary
model.load_state_dict(torch.load('trained_models/trained_IDP-v4_model.pth'))

# Set the model to evaluation mode (important for some models with dropout, batch norm, etc.)
model.eval()

episode_number = 0
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

