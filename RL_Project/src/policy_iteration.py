"""
Auto-generated file
"""

import numpy as np
import random
from src.grid_environment import GridEnvironment
from src.agent import Agent


class PolicyIteration:
    """
    Implements Policy Iteration for solving the GridEnvironment.
    """

    def __init__(self, env, discount_factor=0.9, theta=1e-4):
        """
        Initialize the policy iteration algorithm.

        :param env: The GridEnvironment instance.
        :param discount_factor: Discount factor for future rewards.
        :param theta: Threshold for convergence during policy evaluation.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta

        # Initialize policy and value functions
        self.policy = {
            (r, c): random.choice(list(Agent(env).actions.keys()))
            for r in range(env.rows) for c in range(env.cols)
            if (r, c) != env.goal_state and (r, c) != env.hell_state
        }
        self.value_function = np.zeros((env.rows, env.cols))

    def policy_evaluation(self):
        """
        Perform policy evaluation to compute state-value function.
        """
        while True:
            delta = 0
            new_value_function = np.copy(self.value_function)

            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    state = (r, c)

                    if state == self.env.goal_state or state == self.env.hell_state:
                        continue  # Skip terminal states

                    # Compute value for the state under the current policy
                    action = self.policy[state]
                    move = Agent(self.env).actions[action]
                    next_state = (state[0] + move[0], state[1] + move[1])

                    if not self.env.is_valid_state(next_state):
                        next_state = state  # Agent stays in place if next state is invalid

                    reward = self.env.get_reward(next_state)
                    new_value_function[r, c] = reward + self.discount_factor * self.value_function[next_state]

                    # Update delta for convergence check
                    delta = max(delta, abs(self.value_function[r, c] - new_value_function[r, c]))

            self.value_function = new_value_function
            if delta < self.theta:
                break

    def policy_improvement(self):
        """
        Perform policy improvement to update the policy.
        :return: True if policy is stable, False otherwise.
        """
        policy_stable = True

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)

                if state == self.env.goal_state or state == self.env.hell_state:
                    continue  # Skip terminal states

                # Compute the best action for the current state
                old_action = self.policy[state]
                action_values = {}

                for action, move in Agent(self.env).actions.items():
                    next_state = (state[0] + move[0], state[1] + move[1])

                    if not self.env.is_valid_state(next_state):
                        next_state = state  # Agent stays in place if next state is invalid

                    reward = self.env.get_reward(next_state)
                    action_values[action] = reward + self.discount_factor * self.value_function[next_state]

                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def run_policy_iteration(self):
        """
        Run the policy iteration algorithm.
        :return: Optimized policy and value function.
        """
        iteration = 0
        while True:
            print(f"Policy Iteration Step: {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
            iteration += 1

        return self.policy, self.value_function
