"""
Auto-generated file
"""

import numpy as np
import matplotlib.pyplot as plt
from src.grid_environment import GridEnvironment

def plot_policy(policy, rows, cols):
    """
    Plot the optimal policy on the grid.
    
    :param policy: The optimal policy for each state.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    """
    policy_grid = np.full((rows, cols), "", dtype="str")

    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            if state in policy:
                policy_grid[r, c] = policy[state]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.zeros((rows, cols)), cmap="Blues", alpha=0.1)  # Show grid with transparency
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, policy_grid[r, c], ha="center", va="center", fontsize=12, color="red")
    
    plt.title("Optimal Policy")
    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))
    plt.grid(True)
    plt.show()

def plot_value_function(value_function, rows, cols):
    """
    Plot the value function on the grid.
    
    :param value_function: The value function for each state.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    """
    value_grid = value_function.reshape((rows, cols))

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(value_grid, cmap="YlGnBu", interpolation="nearest")
    fig.colorbar(cax)
    
    plt.title("Optimal Value Function")
    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example: Using the final optimal policy and value function
    env = GridEnvironment(rows=4, cols=3, start_state=(3, 0), goal_state=(0, 2), hell_state=(2, 1))

    # Assume optimal_policy and optimal_value_function are obtained from a previous run (e.g., from GPI)
    optimal_policy = {state: "N" for state in env.get_all_states()}  # Replace with actual policy
    optimal_value_function = np.random.random((env.rows, env.cols))  # Replace with actual value function

    # Visualize the policy and value function
    plot_policy(optimal_policy, env.rows, env.cols)
    plot_value_function(optimal_value_function, env.rows, env.cols)
