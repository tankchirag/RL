import numpy as np

class ValueIteration:
    def __init__(self, env, discount_factor=0.99, theta=1e-6):
        """
        Initialize the Value Iteration algorithm.
        :param env: GridEnvironment instance.
        :param discount_factor: Gamma for discounting future rewards.
        :param theta: Convergence threshold.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.values = np.zeros((env.rows, env.cols))

    def run_value_iteration(self, track_convergence=None):
        """
        Perform Value Iteration.
        :param track_convergence: Optional list to track convergence delta.
        :return: Optimal policy, values, and iteration count.
        """
        iteration = 0
        while True:
            delta = 0
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    state = (row, col)
                    if self.env.is_terminal(state):
                        continue
                    v = self.values[state]
                    self.values[state] = max(
                        sum(prob * (reward + self.discount_factor * self.values[next_state])
                            for prob, next_state, reward in self._get_state_transitions(state, action))
                        for action in self.env.get_possible_actions(state)
                    )
                    delta = max(delta, abs(v - self.values[state]))
            if track_convergence is not None:
                track_convergence.append(delta)
            iteration += 1
            if delta < self.theta:
                break
        policy = self._extract_policy()
        return policy, self.values, iteration

    def _get_state_transitions(self, state, action):
        """
        Get state transitions given a state and action.
        """
        next_state = self.env.get_next_state(state, action)
        reward = self.env.get_reward(next_state)
        return [(1.0, next_state, reward)]  # Deterministic

    def _extract_policy(self):
        """
        Extract the optimal policy from value estimates.
        """
        policy = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                state = (row, col)
                if self.env.is_terminal(state):
                    continue
                policy[state] = max(
                    (action, sum(prob * (reward + self.discount_factor * self.values[next_state])
                                 for prob, next_state, reward in self._get_state_transitions(state, action)))
                    for action in self.env.get_possible_actions(state)
                )[0]
        return policy
