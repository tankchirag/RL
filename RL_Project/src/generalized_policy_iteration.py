"""
Auto-generated file
"""

import numpy as np
from src.grid_environment import GridEnvironment
from src.agent import Agent

class GeneralizedPolicyValueIteration:
    """
    Implements Generalized Policy Iteration (GPI) for solving the GridEnvironment.
    """

    def __init__(self, env, discount_factor=0.9, theta=1e-4, max_iterations=1000):
        """
        Initialize the Generalized Policy Iteration algorithm.

        :param env: The GridEnvironment instance.
        :param discount_factor: Discount factor for future rewards.
        :param theta: Threshold for convergence during value iteration.
        :param max_iterations: Maximum number of iterations to run the algorithm.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations

        # Initialize value function and policy
        self.value_function = np.zeros((env.rows, env.cols))
        self.policy = {state: "N" for state in self.env.get_all_states()}

    def evaluate_policy(self):
        """
        Perform partial policy evaluation by updating the value function for the current policy.
        """
        delta = 0
        new_value_function = np.copy(self.value_function)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)

                if state == self.env.goal_state or state == self.env.hell_state:
                    continue  # Skip terminal states

                # Get the action from the current policy
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
        return delta

    def improve_policy(self):
        """
        Improve the policy by choosing the best action based on the current value function.
        """
        policy_stable = True

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)

                if state == self.env.goal_state or state == self.env.hell_state:
                    continue  # Skip terminal states

                # Calculate the action that maximizes the value function
                action_values = {}
                for action, move in Agent(self.env).actions.items():
                    next_state = (state[0] + move[0], state[1] + move[1])

                    if not self.env.is_valid_state(next_state):
                        next_state = state  # Agent stays in place if next state is invalid

                    reward = self.env.get_reward(next_state)
                    action_values[action] = reward + self.discount_factor * self.value_function[next_state]

                best_action = max(action_values, key=action_values.get)

                # If the best action is not the current action, update the policy
                if self.policy[state] != best_action:
                    policy_stable = False
                    self.policy[state] = best_action

        return policy_stable

    def run_generalized_policy_iteration(self):
        """
        Run the Generalized Policy Iteration algorithm to compute the optimal policy and value function.
        :return: Optimized policy and value function.
        """
        iteration = 0
        while iteration < self.max_iterations:
            print(f"GPI Step: {iteration}")
            delta = self.evaluate_policy()

            if delta < self.theta:
                print("Value function converged.")
                break

            policy_stable = self.improve_policy()

            if policy_stable:
                print("Policy stabilized.")
                break

            iteration += 1

        return self.policy, self.value_function
