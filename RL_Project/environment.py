import numpy as np

class Environment:
    def __init__(self, grid_size, goal_state, hell_state):
        self.grid_size = grid_size
        self.num_states = grid_size[0] * grid_size[1]
        self.actions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # N, NE, E, SE, S, SW, W, NW
        self.goal_state = goal_state
        self.hell_state = hell_state
        self.transition_probs = np.zeros((self.num_states, len(self.actions), self.num_states))
        self.rewards = np.zeros((self.num_states, len(self.actions), self.num_states))

    def _state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def _index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])

    def _calculate_transition_prob(self, state_idx, action_idx, next_state_idx):
        state = self._index_to_state(state_idx)
        next_state = self._index_to_state(next_state_idx)
        action = self.actions[action_idx]

        expected_next_state = (state[0] + action[0], state[1] + action[1])
        if expected_next_state == next_state:
            if 0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]:
                return 1.0
        return 0.0

    def initialize_environment(self):
        goal_index = self._state_to_index(self.goal_state)
        hell_index = self._state_to_index(self.hell_state)

        for state in range(self.num_states):
            for action_idx in range(len(self.actions)):
                for outcome_state in range(self.num_states):
                    self.transition_probs[state, action_idx, outcome_state] = self._calculate_transition_prob(
                        state, action_idx, outcome_state
                    )

        # Set terminal states
        self.transition_probs[goal_index, :, :] = 0
        self.transition_probs[goal_index, :, goal_index] = 1

        self.transition_probs[hell_index, :, :] = 0
        self.transition_probs[hell_index, :, hell_index] = 1

        # Rewards
        self.rewards[:, :, goal_index] = 10
        self.rewards[:, :, hell_index] = -10

