from environment import Environment
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from q_learning import QLearning
from visualization import visualize_environment

def main():
    # Initialize environment
    env = Environment(grid_size=(4, 4), goal_state=(3, 3), hell_state=(2, 2))
    env.initialize_environment()

    # Value Iteration
    value_iteration = ValueIteration(env)
    value_iteration.run()
    print("Optimal Values (Value Iteration):", value_iteration.values)
    print("Optimal Policy (Value Iteration):", value_iteration.policy)
    visualize_environment(env, value_iteration.values, value_iteration.policy)

    # Policy Iteration
    policy_iteration = PolicyIteration(env)
    policy_iteration.run()
    print("Optimal Values (Policy Iteration):", policy_iteration.values)
    print("Optimal Policy (Policy Iteration):", policy_iteration.policy)
    visualize_environment(env, policy_iteration.values, policy_iteration.policy)

    # Q-Learning
    q_learning = QLearning(env)
    q_table = q_learning.train()
    print("Q-Table (Q-Learning):", q_table)

if __name__ == "__main__":
    main()
