import numpy as np

class GridEnvironment:
    def __init__(self, rows, cols, terminal_states, rewards, default_reward=-0.04):
        """
        Initialize the grid environment.
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :param terminal_states: List of terminal state positions (row, col).
        :param rewards: Dictionary mapping positions (row, col) to rewards.
        :param default_reward: Default reward for non-terminal states.
        """
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.default_reward = default_reward
        self.grid = self._initialize_grid()
        self.actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def _initialize_grid(self):
        grid = np.full((self.rows, self.cols), self.default_reward)
        for pos, reward in self.rewards.items():
            grid[pos] = reward
        return grid

    def get_possible_actions(self, state):
        """
        Get possible actions for a given state.
        """
        if state in self.terminal_states:
            return []
        return self.actions

    def get_next_state(self, state, action):
        """
        Get the next state given a state and action.
        :param state: Current position (row, col).
        :param action: Chosen action.
        """
        deltas = {
            'N': (-1, 0), 'NE': (-1, 1), 'E': (0, 1), 'SE': (1, 1),
            'S': (1, 0), 'SW': (1, -1), 'W': (0, -1), 'NW': (-1, -1)
        }
        row, col = state
        dr, dc = deltas[action]
        new_row, new_col = row + dr, col + dc

        # Ensure the new position is within bounds
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            return (new_row, new_col)
        return state  # No movement if out of bounds

    def get_reward(self, state):
        """
        Get the reward for a state.
        """
        return self.rewards.get(state, self.default_reward)

    def is_terminal(self, state):
        """
        Check if a state is terminal.
        """
        return state in self.terminal_states
