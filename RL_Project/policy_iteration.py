class PolicyIteration:
    """
    Implements the Policy Iteration algorithm for policy optimization.
    """

    def __init__(self, env):
        """
        Initialize the Policy Iteration algorithm.

        :param env: An instance of GridEnvironment.
        """
        self.env = env
        self.values = {state: 0 for state in self.env.get_all_states()}
        self.policy = {state: self.env.actions[0] for state in self.env.get_all_states() if not self.env.is_terminal(state)}

    def policy_evaluation(self, epsilon=1e-4):
        """
        Perform policy evaluation to update the value function.

        :param epsilon: Convergence threshold.
        """
        while True:
            delta = 0
            new_values = self.values.copy()

            for state in self.env.get_all_states():
                if self.env.is_terminal(state):
                    continue

                action = self.policy[state]
                q_value = sum(
                    prob * (reward + self.env.gamma * self.values[next_state])
                    for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                )
                delta = max(delta, abs(new_values[state] - q_value))
                new_values[state] = q_value

            self.values = new_values
            if delta < epsilon:
                break

    def policy_improvement(self):
        """
        Perform policy improvement to derive an optimal policy.
        """
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

        return policy_stable

    def run(self):
        """
        Run the Policy Iteration algorithm.

        :return: Tuple (policy, values, iterations).
        """
        iterations = 0

        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
            iterations += 1

        return self.policy, self.values, iterations
