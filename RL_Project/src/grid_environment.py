"""
Auto-generated file
"""

import numpy as np

class GridEnvironment:
    """
    Represents a grid-based stochastic environment for reinforcement learning.
    """

    def __init__(self, rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1), default_reward=-0.04):
        """
        Initialize the grid environment.
        
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :param start_state: Starting position of the agent (row, col).
        :param goal_state: Goal position (row, col).
        :param hell_state: "Hell" position (row, col).
        :param default_reward: Default reward for non-terminal states.
        """
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.goal_state = goal_state
        self.hell_state = hell_state
        self.default_reward = default_reward

        # Initialize the grid with default rewards
        self.grid = np.full((rows, cols), default_reward)
        self.grid[goal_state] = 1  # Goal reward
        self.grid[hell_state] = -1  # Hell penalty

    def is_valid_state(self, state):
        """
        Check if a state is within grid bounds.
        
        :param state: A tuple (row, col).
        :return: True if the state is valid, else False.
        """
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_all_states(self):
            """
            Returns a list of all valid states (excluding goal and hell states).
            """
            all_states = []
            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)
                    if state != self.goal_state and state != self.hell_state:
                        all_states.append(state)
            return all_states

    def get_reward(self, state):
        """
        Get the reward for a specific state.
        
        :param state: A tuple (row, col).
        :return: Reward value.
        """
        if not self.is_valid_state(state):
            raise ValueError("Invalid state.")
        return self.grid[state]

    def print_grid(self):
        """
        Print the grid showing rewards visually.
        """
        print("Environment Grid:")
        print(self.grid)

    def reset(self):
        """
        Reset the grid (for future scalability).
        """
        self.grid = np.full((self.rows, self.cols), self.default_reward)
        self.grid[self.goal_state] = 1
        self.grid[self.hell_state] = -1
