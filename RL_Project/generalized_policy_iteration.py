import numpy as np

class GeneralizedPolicyIteration:
    def __init__(self, env, discount_factor=0.99, theta=1e-6):
        """
        Initialize Generalized Policy Iteration.
        :param env: GridEnvironment instance.
        :param discount_factor: Gamma for discounting future rewards.
        :param theta: Convergence threshold.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.values = np.zeros((env.rows, env.cols))
        self.policy = {state: env.actions[0] for row in range(env.rows) for col in range(env.cols)
                       if not env.is_terminal((row, col))}

    def run_gpi(self):
        """
        Perform Generalized Policy Iteration.
        :return: Optimal policy, values, and iteration count.
        """
        iteration = 0
        while True:
            self._policy_evaluation()
            is_policy_stable = self._policy_improvement()
            iteration += 1
            if is_policy_stable:
                break
        return self.policy, self.values, iteration

    def _policy_evaluation(self):
        """
        Evaluate the current policy.
        """
        while True:
            delta = 0
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    state = (row, col)
                    if self.env.is_terminal(state):
                        continue
                    v = self.values[state]
                    action = self.policy[state]
                    self.values[state] = sum(
                        prob * (reward + self.discount_factor * self.values[next_state])
                        for prob, next_state, reward in self._get_state_transitions(state, action)
                    )
                    delta = max(delta, abs(v - self.values[state]))
            if delta < self.theta:
                break

    def _policy_improvement(self):
        """
        Improve the current policy.
        :return: Whether the policy is stable.
        """
        policy_stable = True
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                state = (row, col)
                if self.env.is_terminal(state):
                    continue
                old_action = self.policy[state]
                self.policy[state] = max(
                    (action, sum(prob * (reward + self.discount_factor * self.values[next_state])
                                 for prob, next_state, reward in self._get_state_transitions(state, action)))
                    for action in self.env.get_possible_actions(state)
                )[0]
                if old_action != self.policy[state]:
                    policy_stable = False
        return policy_stable

    def _get_state_transitions(self, state, action):
        """
        Get state transitions given a state and action.
        """
        next_state = self.env.get_next_state(state, action)
        reward = self.env.get_reward(next_state)
        return [(1.0, next_state, reward)]  # Deterministic
