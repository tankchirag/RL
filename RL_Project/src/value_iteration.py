"""
Auto-generated file
"""

import numpy as np
from src.grid_environment import GridEnvironment
from src.agent import Agent

class ValueIteration:
    """
    Implements Value Iteration for solving the GridEnvironment.
    """

    def __init__(self, env, discount_factor=0.9, theta=1e-4):
        """
        Initialize the value iteration algorithm.

        :param env: The GridEnvironment instance.
        :param discount_factor: Discount factor for future rewards.
        :param theta: Threshold for convergence during value iteration.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta

        # Initialize value function
        self.value_function = np.zeros((env.rows, env.cols))

    def value_update(self):
        """
        Update the value function for all states using the Bellman equation.
        """
        delta = 0
        new_value_function = np.copy(self.value_function)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)

                if state == self.env.goal_state or state == self.env.hell_state:
                    continue  # Skip terminal states

                action_values = []

                # Calculate value for all actions
                for action, move in Agent(self.env).actions.items():
                    next_state = (state[0] + move[0], state[1] + move[1])

                    if not self.env.is_valid_state(next_state):
                        next_state = state  # Agent stays in place if next state is invalid

                    reward = self.env.get_reward(next_state)
                    action_value = reward + self.discount_factor * self.value_function[next_state]
                    action_values.append(action_value)

                # Update value function for the current state
                new_value_function[r, c] = max(action_values)

                # Update delta for convergence check
                delta = max(delta, abs(self.value_function[r, c] - new_value_function[r, c]))

        self.value_function = new_value_function
        return delta

    def extract_policy(self):
        """
        Extract the optimal policy from the value function.
        :return: The optimal policy.
        """
        policy = {}

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)

                if state == self.env.goal_state or state == self.env.hell_state:
                    continue  # Skip terminal states

                # Calculate the best action for the current state
                action_values = {}
                for action, move in Agent(self.env).actions.items():
                    next_state = (state[0] + move[0], state[1] + move[1])

                    if not self.env.is_valid_state(next_state):
                        next_state = state  # Agent stays in place if next state is invalid

                    reward = self.env.get_reward(next_state)
                    action_values[action] = reward + self.discount_factor * self.value_function[next_state]

                best_action = max(action_values, key=action_values.get)
                policy[state] = best_action

        return policy

    # def run_value_iteration(self):
    #     """
    #     Run the value iteration algorithm to compute the optimal policy and value function.
    #     :return: Optimized policy and value function.
    #     """
    #     iteration = 0
    #     while True:
    #         print(f"Value Iteration Step: {iteration}")
    #         delta = self.value_update()
    #         if delta < self.theta:
    #             break
    #         iteration += 1

    #     optimal_policy = self.extract_policy()
    #     return optimal_policy, self.value_function, iteration

    def run_value_iteration(self, epsilon=1e-4):
        iterations = 0  # Track iterations
        while True:
            delta = 0
            for state in self.env.get_all_states():
                if self.env.is_terminal(state):
                    continue

                max_value = float('-inf')
                for action in self.env.actions:
                    value = sum(
                        prob * (reward + self.env.gamma * self.value_function[next_state])
                        for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                    )
                    max_value = max(max_value, value)

                delta = max(delta, abs(max_value - self.value_function[state]))
                self.value_function[state] = max_value

            iterations += 1  # Increment iteration count
            if delta < epsilon:
                break

        for state in self.env.get_all_states():
            if self.env.is_terminal(state):
                continue

            best_action = max(
                self.env.actions,
                key=lambda action: sum(
                    prob * (reward + self.env.gamma * self.value_function[next_state])
                    for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                ),
            )
            self.policy[state] = best_action

        return self.policy, self.value_function, iterations
