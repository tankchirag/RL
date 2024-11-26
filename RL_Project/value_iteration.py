import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=0.99, threshold=1e-6):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.values = np.zeros(env.num_states)  # Corrected to match 'num_states'
        self.policy = np.zeros(env.num_states, dtype=int)  # Corrected to match 'num_states'

    def run(self):
        while True:
            delta = 0  # Tracks convergence
            new_values = np.copy(self.values)

            for s in range(self.env.num_states):  # Corrected to match 'num_states'
                action_values = []
                for a in range(len(self.env.actions)):
                    action_value = sum(
                        self.env.transition_probs[s, a, s_prime] *
                        (self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime])
                        for s_prime in range(self.env.num_states)
                    )
                    action_values.append(action_value)

                # Update values and determine best action
                best_action_value = max(action_values)
                best_action = np.argmax(action_values)
                new_values[s] = best_action_value
                self.policy[s] = best_action  # Update the policy
                delta = max(delta, abs(best_action_value - self.values[s]))

            self.values = new_values

            # Convergence check
            if delta < self.threshold:
                break
