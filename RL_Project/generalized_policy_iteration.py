class GeneralizedPolicyIteration:
    """
    Combines Policy Iteration and Value Iteration for generalized optimization.
    """

    def __init__(self, env):
        """
        Initialize the Generalized Policy Iteration algorithm.

        :param env: An instance of GridEnvironment.
        """
        self.env = env
        self.values = {state: 0 for state in self.env.get_all_states()}
        self.policy = {state: self.env.actions[0] for state in self.env.get_all_states() if not self.env.is_terminal(state)}

    def iterate(self, policy_eval_steps=3, epsilon=1e-4):
        """
        Perform generalized policy iteration.

        :param policy_eval_steps: Number of policy evaluation iterations.
        :param epsilon: Convergence threshold.
        :return: Tuple (policy, values, iterations).
        """
        iterations = 0

        while True:
            # Policy Evaluation (limited steps)
            for _ in range(policy_eval_steps):
                new_values = self.values.copy()
                for state in self.env.get_all_states():
                    if self.env.is_terminal(state):
                        continue

                    action = self.policy[state]
                    q_value = sum(
                        prob * (reward + self.env.gamma * self.values[next_state])
                        for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                    )
                    new_values[state] = q_value

                self.values = new_values

            # Policy Improvement
            policy_stable = True
            for state in self.env.get_all_states():
                if self.env.is_terminal(state):
                    continue

                old_action = self.policy[state]
                self.policy[state] = max(
                    self.env.actions,
                    key=lambda action: sum(
                        prob * (reward + self.env.gamma * self.values[next_state])
                        for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                    ),
                )
                if old_action != self.policy[state]:
                    policy_stable = False

            iterations += 1
            if policy_stable:
                break

        return self.policy, self.values, iterations
