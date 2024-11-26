import numpy as np

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((env.n_states, env.n_actions))

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False

            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.env.n_actions)  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit

                next_state, reward, done = self.env.step(state, action)
                best_next_action = np.argmax(self.q_table[next_state])

                # Q-Learning update
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
                )

                state = next_state

        return self.q_table
