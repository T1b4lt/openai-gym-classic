class Discretizator:
    def __init__(self, low, high, bins_pos, bins_vel):
        self.pos_min_value = low[0]
        self.pos_max_value = high[0]
        self.vel_min_value = low[1]
        self.vel_max_value = high[1]
        self.pos_n_bins = bins_pos
        self.vel_n_bins = bins_vel
        self.pos_bin_size = (self.pos_max_value -
                             self.pos_min_value) / self.pos_n_bins
        self.vel_bin_size = (self.vel_max_value -
                             self.vel_min_value) / self.vel_n_bins

    def n_states(self):
        return self.pos_n_bins * self.vel_n_bins

    def idx_state(self, state):
        value_pos = state[0]
        value_vel = state[1]
        idx_pos = self.discretize_pos(value_pos)
        idx_vel = self.discretize_vel(value_vel)
        return idx_pos * self.vel_n_bins + idx_vel

    def discretize_pos(self, value):
        if value < self.pos_min_value:
            return 0
        elif value > self.pos_max_value:
            return self.pos_n_bins - 1
        else:
            return int((value - self.pos_min_value) / self.pos_bin_size)

    def discretize_vel(self, value):
        if value < self.vel_min_value:
            return 0
        elif value > self.vel_max_value:
            return self.vel_n_bins - 1
        else:
            return int((value - self.vel_min_value) / self.vel_bin_size)
