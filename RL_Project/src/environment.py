import numpy as np

class GridEnvironment:
    """
    Represents the grid environment for reinforcement learning with customizable rewards and actions.
    """

    def __init__(self, rows, cols, terminal_states, rewards, gamma=0.9):
        """
        Initialize the grid environment.

        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :param terminal_states: List of terminal states (row, col).
        :param rewards: Dictionary with (row, col) as keys and rewards as values.
        :param gamma: Discount factor.
        """
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.gamma = gamma
        self.actions = [
            (-1, 0),  # North
            (1, 0),   # South
            (0, -1),  # West
            (0, 1),   # East
            (-1, 1),  # North-East
            (-1, -1), # North-West
            (1, 1),   # South-East
            (1, -1)   # South-West
        ]
        self.grid = self._initialize_grid()

    def _initialize_grid(self):
        """
        Initialize the grid with default rewards and terminal states.

        :return: The grid as a 2D numpy array.
        """
        grid = np.full((self.rows, self.cols), -0.04)  # Default reward
        for state, reward in self.rewards.items():
            grid[state] = reward
        return grid

    def is_terminal(self, state):
        """
        Check if a state is terminal.

        :param state: Tuple (row, col).
        :return: True if the state is terminal, False otherwise.
        """
        return state in self.terminal_states

    def get_all_states(self):
        """
        Get all possible states in the grid.

        :return: List of (row, col) tuples.
        """
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def get_next_states_and_rewards(self, state, action):
        """
        Get the next states, rewards, and probabilities for a given state and action.

        :param state: Current state (row, col).
        :param action: Action to perform (delta_row, delta_col).
        :return: List of (next_state, reward, probability) tuples.
        """
        if self.is_terminal(state):
            return [(state, 0, 1)]  # Terminal state stays the same

        next_state = (state[0] + action[0], state[1] + action[1])
        if not (0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.cols):
            next_state = state  # Out-of-bounds actions result in staying in place

        reward = self.grid[next_state]
        return [(next_state, reward, 1.0)]  # Single deterministic outcome
