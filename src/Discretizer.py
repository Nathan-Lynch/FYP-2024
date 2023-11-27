import numpy as np

class Discretizer:
    def __init__(self, action_space, num_bins):
        self.action_space = action_space
        self.num_bins = num_bins
        self.bins = []
        for dim, bins in zip(action_space, num_bins):
            self.bins.append(np.linspace(min(dim), max(dim), bins))

    def discretize(self, action):
        discrete_action = []
        for i in range(len(action)):
            idx = int(np.digitize(action[i], self.bins[i]) - 1)
            discrete_action.append(idx)
        return tuple(discrete_action)

    def undiscretize(self, discrete_action):
        if not isinstance(discrete_action, (tuple, list)):
            discrete_action = (discrete_action,)

        continuous_action = []
        for i in range(len(discrete_action)):
            idx = int(discrete_action[i])
            min_val, max_val = self.bins[i][idx], self.bins[i][idx + 1]
            value = (min_val + max_val) / 2.0
            continuous_action.append(value)

        return continuous_action