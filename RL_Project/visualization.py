import matplotlib.pyplot as plt
import numpy as np

def plot_policy(policy, rows, cols, title="Policy"):
    grid = np.full((rows, cols), '', dtype=object)
    for state, action in policy.items():
        grid[state] = action
    plt.figure(figsize=(8, 6))
    plt.table(cellText=grid, loc='center', cellLoc='center')
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_convergence(track_convergence, method):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(track_convergence)), track_convergence, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Delta")
    plt.title(f"{method} Convergence")
    plt.grid()
    plt.show()
