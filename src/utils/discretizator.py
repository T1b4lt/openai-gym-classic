import numpy as np


class Discretizator:
    def __init__(self, low_array, high_array, bins_array):
        self.low_array = np.array(low_array)
        self.high_array = np.array(high_array)
        self.bins_array = np.array(bins_array)
        self.bin_size_array = (
            self.high_array - self.low_array) / self.bins_array

    def get_n_states(self):
        return np.prod(self.bins_array)

    def idx_state(self, state):
        value_array = []
        for i in range(len(state)):
            value_array.append(self.discretize_value(state[i], i))
        return np.ravel_multi_index(value_array, self.bins_array)

    def discretize_value(self, value, i):
        if value < self.low_array[i]:
            return 0
        elif value >= self.high_array[i]:
            return self.bins_array[i] - 1
        else:
            return int((value - self.low_array[i]) / self.bin_size_array[i])
