import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    ''' Deep Q-Network, calculates Q-values and outputs them'''
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) # input: n_observations, output: 128
        self.layer2 = nn.Linear(128, 128) # input: 128, output: 128
        self.layer3 = nn.Linear(128, n_actions) # input: 128, output: n_actions

    def forward(self, x):
        ''' Computes Q-values for a given input state.
            Called with either one element to determine next action, or a 
            batch during optimization.
            Returns tensor([[left0exp,right0exp]...])'''
        # ReLU(x) = max(0, x), leaving positive values unchanged
        x = F.relu(self.layer1(x)) # input: x, ReLU applied to output
        x = F.relu(self.layer2(x)) # input: output of layer1
        # output is used to estimate expected future rewards
        return self.layer3(x) # input: output of layer2, Q-values for each action are returned as a tensor of shape (batch_size, n_actions)