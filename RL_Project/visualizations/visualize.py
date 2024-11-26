import matplotlib.pyplot as plt
import numpy as np

def visualize_policy(env, policy, values):
    rows, cols = env.rows, env.cols
    policy_arrows = np.array(['↑', '←', '↓', '→', '↖', '↗', '↙', '↘'])
    grid = np.array(policy_arrows[policy]).reshape(rows, cols)
    value_grid = values.reshape(rows, cols)

    plt.figure(figsize=(8, 8))
    plt.title("Policy Visualization")
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, f"{grid[i, j]}\n{value_grid[i, j]:.1f}", ha='center', va='center', fontsize=12)

    plt.xticks(range(cols), [])
    plt.yticks(range(rows), [])
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
