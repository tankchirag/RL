class ValueIteration:
    """
    Implements the Value Iteration algorithm for policy optimization.
    """

    def __init__(self, env):
        """
        Initialize the Value Iteration algorithm.

        :param env: An instance of GridEnvironment.
        """
        self.env = env
        self.values = {state: 0 for state in self.env.get_all_states()}
        self.policy = {state: None for state in self.env.get_all_states()}

    def run(self, epsilon=1e-4):
        """
        Run the Value Iteration algorithm.

        :param epsilon: Convergence threshold.
        :return: Tuple (policy, values, iterations).
        """
        iterations = 0
        while True:
            delta = 0
            new_values = self.values.copy()

            for state in self.env.get_all_states():
                if self.env.is_terminal(state):
                    continue

                action_values = []
                for action in self.env.actions:
                    q_value = sum(
                        prob * (reward + self.env.gamma * self.values[next_state])
                        for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                    )
                    action_values.append(q_value)

                max_value = max(action_values)
                delta = max(delta, abs(new_values[state] - max_value))
                new_values[state] = max_value

            self.values = new_values
            iterations += 1
            if delta < epsilon:
                break

        # Derive policy from value function
        for state in self.env.get_all_states():
            if self.env.is_terminal(state):
                continue

            self.policy[state] = max(
                self.env.actions,
                key=lambda action: sum(
                    prob * (reward + self.env.gamma * self.values[next_state])
                    for next_state, reward, prob in self.env.get_next_states_and_rewards(state, action)
                ),
            )

        return self.policy, self.values, iterations
