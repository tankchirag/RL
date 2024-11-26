import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=0.9, threshold=1e-4):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.values = np.zeros(env.num_states)

    def run(self):
        while True:
            delta = 0
            new_values = np.copy(self.values)

            for s in range(self.env.num_states):
                q_values = np.zeros(len(self.env.actions))
                for a in range(len(self.env.actions)):
                    q_values[a] = sum(
                        self.env.transition_probs[s, a, s_prime] * (
                            self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime]
                        ) for s_prime in range(self.env.num_states)
                    )
                new_values[s] = max(q_values)
                delta = max(delta, abs(new_values[s] - self.values[s]))

            self.values = new_values
            if delta < self.threshold:
                break

        policy = np.zeros(self.env.num_states, dtype=int)
        for s in range(self.env.num_states):
            q_values = np.zeros(len(self.env.actions))
            for a in range(len(self.env.actions)):
                q_values[a] = sum(
                    self.env.transition_probs[s, a, s_prime] * (
                        self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime]
                    ) for s_prime in range(self.env.num_states)
                )
            policy[s] = np.argmax(q_values)

        return self.values, policy
# import numpy as np

# class ValueIteration:
#     def __init__(self, env, gamma=0.99, theta=1e-6):
#         self.env = env
#         self.gamma = gamma
#         self.theta = theta
#         self.values = np.zeros(env.n_states)
#         self.policy = np.zeros(env.n_states, dtype=int)  # Initialize policy array

#     def run(self):
#         while True:
#             delta = 0
#             new_values = np.copy(self.values)

#             for s in range(self.env.n_states):
#                 action_values = []
#                 for a in range(self.env.n_actions):
#                     action_value = sum(
#                         self.env.transition_probs[s, a, s_prime] *
#                         (self.env.rewards[s, a, s_prime] + self.gamma * self.values[s_prime])
#                         for s_prime in range(self.env.n_states)
#                     )
#                     action_values.append(action_value)

#                 best_action_value = max(action_values)
#                 new_values[s] = best_action_value
#                 best_action = np.argmax(action_values)
#                 self.policy[s] = best_action  # Update the policy
#                 delta = max(delta, abs(best_action_value - self.values[s]))

#             self.values = new_values
#             if delta < self.theta:
#                 break
