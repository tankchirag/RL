import time
import numpy as np
import matplotlib.pyplot as plt
from src.grid_environment import GridEnvironment
from src.policy_iteration import PolicyIteration
from src.value_iteration import ValueIteration
from src.generalized_policy_iteration import GeneralizedPolicyValueIteration

def evaluate_method(method_class, env, discount_factor=0.9, max_iterations=1000, theta=1e-4):
    """
    Evaluate the performance of a given method (Policy Iteration, Value Iteration, or GPI).
    
    :param method_class: Class implementing the method (PolicyIteration, ValueIteration, or GPI).
    :param env: The environment object (GridEnvironment).
    :param discount_factor: Discount factor for future rewards.
    :param max_iterations: Maximum iterations for convergence.
    :param theta: Threshold for convergence.
    
    :return: Optimal policy, optimal value function, number of iterations, and execution time.
    """
    start_time = time.time()

    method = method_class(env, discount_factor=discount_factor, max_iterations=max_iterations, theta=theta)
    optimal_policy, optimal_value_function = method.run_policy_iteration()

    end_time = time.time()
    execution_time = end_time - start_time

    return optimal_policy, optimal_value_function, method.iteration_count, execution_time

def plot_comparison(steps, times, labels):
    """
    Plot the comparison of convergence steps and execution times for different methods.
    
    :param steps: List of steps (number of iterations) for each method.
    :param times: List of execution times for each method.
    :param labels: Labels for each method to display in the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot number of steps (iterations)
    ax[0].bar(labels, steps)
    ax[0].set_title("Number of Iterations to Convergence")
    ax[0].set_ylabel("Iterations")

    # Plot execution times
    ax[1].bar(labels, times)
    ax[1].set_title("Execution Time for Convergence")
    ax[1].set_ylabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the environment
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Evaluate Policy Iteration
    pi_policy, pi_value, pi_steps, pi_time = evaluate_method(PolicyIteration, env)

    # Evaluate Value Iteration
    vi_policy, vi_value, vi_steps, vi_time = evaluate_method(ValueIteration, env)

    # Evaluate Generalized Policy Iteration
    gpi_policy, gpi_value, gpi_steps, gpi_time = evaluate_method(GeneralizedPolicyValueIteration, env)

    # Compare results
    print(f"Policy Iteration: {pi_steps} steps, {pi_time:.4f} seconds")
    print(f"Value Iteration: {vi_steps} steps, {vi_time:.4f} seconds")
    print(f"Generalized Policy Iteration: {gpi_steps} steps, {gpi_time:.4f} seconds")

    # Plot the comparison
    steps = [pi_steps, vi_steps, gpi_steps]
    times = [pi_time, vi_time, gpi_time]
    labels = ["Policy Iteration", "Value Iteration", "GPI"]
    plot_comparison(steps, times, labels)
