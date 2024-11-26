import numpy as np

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((env.num_states, env.num_actions))  # Changed to num_states

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)  # Explore: Random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: Best action based on Q-table

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def run(self):
        for episode in range(self.episodes):
            state = self.env.reset()  # Reset the environment to the starting state
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
