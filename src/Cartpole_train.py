import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from plots import plot_rewards_loss

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym.make("CartPole-v1")

#plt.ion() # enables interactive plot

# Checks if cuda compatible GPU is available, uses CPU if not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creates a tuple that can be accessed by name instead of index
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    ''' Allows us to create instances of a replay buffer'''
    def __init__(self, capacity):
        ''' Memory is implemented as a double ended queue with a max length of 'capacity'.
            This ensures that the buffer wont exceed the specified capacity, and when
            new transitions are added beyond the capacity, the oldest transitions will
            be removed. '''
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        ''' Sample a batch of transitions from the replay buffer for training.
            It takes a batch_size arg, which specifies how many transitions to sample.'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' Returns current number of transitions.'''
        return len(self.memory)
    
BATCH_SIZE = 128 # No. of transitions sampled from the replay buffer
GAMMA = 0.99 # The discount factor
EPS_START = 0.9 # The starting value of epsilon
EPS_END = 0.05 # The final value of epsilon
EPS_DECAY = 1000 # Controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # The update rate of the target network
LR = 1e-4 # Learning rate of the `AdamW` optimizer

state, info = env.reset()
n_observations = len(state) # Get the number of state observations
n_actions = env.action_space.n # Get number of actions from gym action space

# Instances of DQN, used to approx Q-values for the state-action pairs
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
# target_net loaded with weights of policy_net, maintains a target
# network with slowly moving weights, stabilizes training.
target_net.load_state_dict(policy_net.state_dict())

# Optimizes the params of policy_net with learning rate LR, and uses the
# AMSGrad varient of the Adam optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000) # Instance of repaly memory with 10000 transitions as capacity

steps_done = 0 # tracks steps taken by agent

def select_action(state):
    ''' Selects action for the agent using epsilon-greedy algorithm'''
    global steps_done
    sample = random.random() # sample between 0 and 1

    # Determines probability of exploration vs exploitation
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1

    # If sample > eps_threshold, agent selects action with the max expected reward
    # based on the Q-values from policy_net
    # Otherwise select a random action
    if sample > eps_threshold:
        # turns off gradient tracking in Pytorch as we arent updating model weights.
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1) # Selects action with max Q-value
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # Selects a random action

episode_durations = []
rewards = []
losses = []

def optimize_model():
    ''' Trains DQN  model using the collected experiences stores in the 
        replay buffer.'''
    
    # If the number of transitions less than batch size, training not performed
    if len(memory) < BATCH_SIZE:
        return 0
    
    # Gets a sample of transitions from replay buffer
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net.
    # Calculated using state_batch through policy_net, then gathers the Q-values
    # corresponding to the action taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values, uses bellman equation
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss, computes the loss between the predicted Q-values
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model, model gradients are zeroed, resetting the gradient
    # accumulations from the previous iterations.
    optimizer.zero_grad()

    # Loss is backpropogated to compute the gradients with respect to the models parameters
    loss.backward()

    # In-place gradient clipping, prevents large gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    # Model parameters are updated using a gradient descent step.
    optimizer.step()

    return loss.item()

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 1000

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for i_episode in range(num_episodes):

    # Initialize the environment and get it's state
    state, info = env.reset()

    # converts state to a tensor
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    total_loss_per_episode = 0  # Initialize total_loss_per_episode

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory (replay buffer)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        total_loss_per_episode += loss  # Accumulate loss for the episode

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            rewards.append(total_reward)
            #plot_rewards_loss(axs, rewards, losses)
            break
        
        total_reward += reward.item()
    
    avg_loss_per_episode = total_loss_per_episode / (t + 1)  # Divide by the total number of steps in the episode
    losses.append(avg_loss_per_episode)

torch.save(policy_net.state_dict(), 'C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_cartpole_model.pth')

print('Complete')
plot_rewards_loss(axs, rewards, losses, show_result=True)
#plt.ioff()
#plt.show()