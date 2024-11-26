import numpy as np

class GridEnvironment:
    def __init__(self, rows, cols, terminal_states, rewards):
        """
        Initialize the grid environment.
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :param terminal_states: List of terminal states as tuples [(row, col), ...].
        :param rewards: Dictionary of rewards for each state {state: reward}.
        """
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']  # 8 possible actions

    def is_terminal(self, state):
        """
        Check if the state is terminal.
        :param state: Tuple (row, col).
        :return: True if terminal, False otherwise.
        """
        return state in self.terminal_states

    def get_all_states(self):
        """
        Get a list of all possible states in the grid.
        :return: List of all states as tuples [(row, col), ...].
        """
        return [(row, col) for row in range(self.rows) for col in range(self.cols)]

    def get_possible_actions(self, state):
        """
        Get possible actions from a given state.
        :param state: Current state (row, col).
        :return: List of possible actions.
        """
        if self.is_terminal(state):
            return []  # No actions possible in terminal state
        return self.actions

    def get_next_state(self, state, action):
        """
        Determine the next state based on the current state and action.
        :param state: Current state (row, col).
        :param action: Action to take.
        :return: Next state as a tuple (row, col).
        """
        row, col = state
        if action == 'N':
            row = max(0, row - 1)
        elif action == 'NE':
            row, col = max(0, row - 1), min(self.cols - 1, col + 1)
        elif action == 'E':
            col = min(self.cols - 1, col + 1)
        elif action == 'SE':
            row, col = min(self.rows - 1, row + 1), min(self.cols - 1, col + 1)
        elif action == 'S':
            row = min(self.rows - 1, row + 1)
        elif action == 'SW':
            row, col = min(self.rows - 1, row + 1), max(0, col - 1)
        elif action == 'W':
            col = max(0, col - 1)
        elif action == 'NW':
            row, col = max(0, row - 1), max(0, col - 1)
        return (row, col)

    def get_reward(self, state):
        """
        Get the reward for a given state.
        :param state: State tuple (row, col).
        :return: Reward for the state.
        """
        return self.rewards.get(state, -0.04)  # Default reward is -0.04
