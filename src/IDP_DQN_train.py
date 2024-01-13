import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from Discretizer import Discretizer
import os

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

env = gym.make("InvertedDoublePendulum-v4")

# set up matplotlib, checks if using ipython backend
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() # enables interactive plot

# Checks if cuda compatible GPU is available, uses CPU if not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creates a tuple that can be accessed by name instead of index
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    ''' Allows us to create instances of a replay buffer'''
    def __init__(self, capacity):
        ''' Memory is implemented as a double ended queue) with a max length of 'capacity'.
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
    
BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.05 # EPS_END is the final value of epsilon
EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 1e-4 # LR is the learning rate of the `AdamW` optimizer

state, info = env.reset()
n_observations = len(state)  # Get the number of state observations
n_actions = env.action_space.shape[0]

# Adjust the number of bins as needed
num_bins = [5]  # Since you have one-dimensional action space
action_space = [np.linspace(min_action, max_action, num_bin) for min_action, max_action, num_bin in zip(env.action_space.low, env.action_space.high, num_bins)]
discretizer = Discretizer(action_space, num_bins)

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
    sample = random.random()  # sample between 0 and 1

    # Determines probability of exploration vs exploitation
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1

    # If sample > eps_threshold, agent selects action with the max expected reward
    # based on the Q-values from policy_net
    # Otherwise, select a random action
    if sample > eps_threshold:
        with torch.no_grad():
            # Discretize the continuous action before using it
            continuous_action = policy_net(state).max(1)[1].view(1, 1).item()
            # Use [0] to get the first element of the tuple returned by discretizer.discretize
            action = torch.tensor([discretizer.discretize([continuous_action])[0]], device=device, dtype=torch.float32)
        return action
    else:
        # Discretize the random continuous action before using it
        random_continuous_action = env.action_space.sample()
        # Use [0] to get the first element of the tuple returned by discretizer.discretize
        action = torch.tensor([discretizer.discretize([random_continuous_action])[0]], device=device, dtype=torch.float32)
        return action

# List to store duration of each episode while training
episode_durations = []

def plot_durations(show_result=False):
    ''' Takes optional Bool parameter show_result that determines whether the plot
        indicates that its displaying results or during training.'''
    
    plt.figure(1) # creare plot with figure 1

    # episode durations converted into a tensor with float data type
    durations_t = torch.tensor(episode_durations, dtype=torch.float) 

    # If not training
    if show_result:
        plt.title('Result')
    # if training
    else:
        plt.clf() # clear figure, resets to empty plot
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy()) # plots duration of each episode

    # Take 100 episode averages and plot them
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            # Get current figure, returns reference to the current figure
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    ''' Trains DQN model using the collected experiences stored in the 
        replay buffer.'''
    
    # If the number of transitions is less than batch size, training is not performed
    if len(memory) < BATCH_SIZE:
        return
    
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
    action_batch = torch.cat(batch.action).long()
    action_batch = action_batch.view(-1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net.
    # Calculated using state_batch through policy_net, then gathers the Q-values
    # corresponding to the action taken
    policy_output = policy_net(state_batch).view(-1)
    state_action_values = policy_output.gather(0, action_batch)
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

    # Loss is backpropagated to compute the gradients with respect to the models parameters
    loss.backward()

    # In-place gradient clipping, prevents large gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    # Model parameters are updated using a gradient descent step.
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 100000
else:
    num_episodes = 50

for i_episode in range(num_episodes):

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(discretizer.undiscretize(action.item()))
        reward = torch.tensor([reward], device=device)

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

torch.save(policy_net.state_dict(), 'C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_IDP_DQN_model.pth')

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()