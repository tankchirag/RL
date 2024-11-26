import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = np.zeros(env.num_states, dtype=int)
        self.values = np.zeros(env.num_states)

    def evaluate_policy(self):
        while True:
            delta = 0
            new_values = np.copy(self.values)

            for s in range(self.env.num_states):
                a = self.policy[s]
                new_values[s] = sum(
                    self.env.transition_probs[s, a, s_prime] * (
                        self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime]
                    ) for s_prime in range(self.env.num_states)
                )
                delta = max(delta, abs(new_values[s] - self.values[s]))

            self.values = new_values
            if delta < 1e-4:
                break

    def improve_policy(self):
        stable = True

        for s in range(self.env.num_states):
            old_action = self.policy[s]
            q_values = np.zeros(len(self.env.actions))

            for a in range(len(self.env.actions)):
                q_values[a] = sum(
                    self.env.transition_probs[s, a, s_prime] * (
                        self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime]
                    ) for s_prime in range(self.env.num_states)
                )

            self.policy[s] = np.argmax(q_values)
            if old_action != self.policy[s]:
                stable = False

        return stable

    def run(self):
        while True:
            self.evaluate_policy()
            if self.improve_policy():
                break

        return self.values, self.policy
