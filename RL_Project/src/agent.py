"""
Auto-generated file
"""

import numpy as np
import random

class Agent:
    """
    Represents the agent in the grid environment.
    """

    def __init__(self, env, stochastic_prob=0.1):
        """
        Initialize the agent.

        :param env: The GridEnvironment instance.
        :param stochastic_prob: Probability of taking a random action instead of the intended one.
        """
        self.env = env
        self.position = env.start_state  # Initialize agent at the start state
        self.stochastic_prob = stochastic_prob

        # Define possible actions
        self.actions = {
            "N": (-1, 0),  # North
            "S": (1, 0),   # South
            "E": (0, 1),   # East
            "W": (0, -1),  # West
            "NE": (-1, 1), # North-East
            "NW": (-1, -1),# North-West
            "SE": (1, 1),  # South-East
            "SW": (1, -1), # South-West
        }

    def reset(self):
        """
        Reset the agent to the start state.
        """
        self.position = self.env.start_state

    def step(self, action):
        """
        Take a step in the environment.

        :param action: Action to take (e.g., "N", "S").
        :return: New position, reward, and whether the episode is done.
        """
        if random.random() < self.stochastic_prob:
            action = random.choice(list(self.actions.keys()))  # Stochastic behavior
        
        # Calculate the new position
        move = self.actions.get(action, (0, 0))  # Default to no movement if invalid action
        new_position = (self.position[0] + move[0], self.position[1] + move[1])

        # Check if the new position is valid
        if self.env.is_valid_state(new_position):
            self.position = new_position

        # Get the reward and check if the episode is done
        reward = self.env.get_reward(self.position)
        done = self.position == self.env.goal_state or self.position == self.env.hell_state

        return self.position, reward, done

    def get_possible_actions(self):
        """
        Get all possible actions for the agent.
        :return: List of action keys (e.g., ["N", "S", "E", "W", ...]).
        """
        return list(self.actions.keys())
