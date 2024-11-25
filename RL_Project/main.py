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
    optimal_policy, optimal_values, iteration_pi = policy_iter.run_policy_iteration()

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


import time
from src.grid_environment import GridEnvironment
from src.policy_iteration import PolicyIteration
from src.value_iteration import ValueIteration
from src.generalized_policy_iteration import GeneralizedPolicyValueIteration
import matplotlib.pyplot as plt
import numpy as np

def compare_methods(env):
    """
    Run and compare Policy Iteration, Value Iteration, and Generalized Policy Iteration.
    :param env: The GridEnvironment instance.
    :return: Comparison data for analysis and visualization.
    """
    results = {}

    # Policy Iteration
    print("\n--- Running Policy Iteration ---")
    start_time = time.time()
    pi = PolicyIteration(env)
    policy_pi, values_pi, iterations_pi = pi.run_policy_iteration()
    time_pi = time.time() - start_time
    results["Policy Iteration"] = {
        "policy": policy_pi,
        "values": values_pi,
        "iterations": iterations_pi,
        "time": time_pi,
    }

    # Value Iteration
    print("\n--- Running Value Iteration ---")
    start_time = time.time()
    vi = ValueIteration(env)
    policy_vi, values_vi, iterations_vi = vi.run_value_iteration()
    time_vi = time.time() - start_time
    results["Value Iteration"] = {
        "policy": policy_vi,
        "values": values_vi,
        "iterations": iterations_vi,
        "time": time_vi,
    }

    # Generalized Policy Iteration
    print("\n--- Running Generalized Policy Iteration ---")
    start_time = time.time()
    gpi = GeneralizedPolicyValueIteration(env)
    policy_gpi, values_gpi = gpi.run_generalized_policy_iteration()
    time_gpi = time.time() - start_time
    results["Generalized Policy Iteration"] = {
        "policy": policy_gpi,
        "values": values_gpi,
        "iterations": "Flexible",  # GPI doesn't have fixed iteration steps
        "time": time_gpi,
    }

    return results

def visualize_results(env, results):
    """
    Visualize the results of Policy Iteration, Value Iteration, and Generalized Policy Iteration.
    :param env: The GridEnvironment instance.
    :param results: Dictionary containing results of all methods.
    """
    methods = results.keys()

    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    for idx, method in enumerate(methods):
        values = results[method]["values"]
        ax = axes[idx]
        ax.matshow(values, cmap="coolwarm", alpha=0.8)
        for (i, j), val in np.ndenumerate(values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
        ax.set_title(f"{method}\nTime: {results[method]['time']:.2f}s")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the environment
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Compare methods
    results = compare_methods(env)

    # Print Results
    for method, data in results.items():
        print(f"\n--- {method} ---")
        print(f"Iterations: {data['iterations']}")
        print(f"Time Taken: {data['time']:.2f}s")
        print("Optimal Value Function:")
        print(data["values"])

    # Visualize Results
    visualize_results(env, results)
