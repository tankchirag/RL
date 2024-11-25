"""
Auto-generated file
"""
from src.grid_environment import GridEnvironment
from src.policy_iteration import PolicyIteration
from src.agent import Agent
import random



if __name__ == "__main__":
    # Initialize environment and agent
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))
    agent = Agent(env, stochastic_prob=0.1)

    # Print the environment
    env.print_grid()

    # Simulate one episode
    agent.reset()
    print(f"Agent starts at: {agent.position}")

    done = False
    while not done:
        action = random.choice(agent.get_possible_actions())
        new_position, reward, done = agent.step(action)
        print(f"Agent moves {action} to {new_position} with reward {reward}")
    
    print("Episode finished!")




if __name__ == "__main__":
    # Initialize the environment
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Run Policy Iteration
    policy_iter = PolicyIteration(env)
    optimal_policy, optimal_values = policy_iter.run_policy_iteration()

    # Display results
    print("Optimized Policy:")
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in policy_iter.policy:
                print(f"State {state}: {policy_iter.policy[state]}")
            else:
                print(f"State {state}: Terminal")

    print("\nOptimized Value Function:")
    print(optimal_values)


from src.grid_environment import GridEnvironment
from src.value_iteration import ValueIteration

if __name__ == "__main__":
    # Initialize the environment
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Run Value Iteration
    value_iter = ValueIteration(env)
    optimal_policy, optimal_values = value_iter.run_value_iteration()

    # Display results
    print("Optimized Policy:")
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in optimal_policy:
                print(f"State {state}: {optimal_policy[state]}")
            else:
                print(f"State {state}: Terminal")

    print("\nOptimized Value Function:")
    print(optimal_values)


from src.grid_environment import GridEnvironment
from src.generalized_policy_iteration import GeneralizedPolicyValueIteration

if __name__ == "__main__":
    # Initialize the environment
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Run Generalized Policy Iteration
    gpi = GeneralizedPolicyValueIteration(env)
    optimal_policy, optimal_values = gpi.run_generalized_policy_iteration()

    # Display results
    print("Optimized Policy:")
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in optimal_policy:
                print(f"State {state}: {optimal_policy[state]}")
            else:
                print(f"State {state}: Terminal")

    print("\nOptimized Value Function:")
    print(optimal_values)
