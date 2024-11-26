import matplotlib.pyplot as plt
import numpy as np

def visualize_environment(env, values, policy):
    grid_size = env.grid_size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid
    for x in range(grid_size[0] + 1):
        ax.plot([0, grid_size[1]], [x, x], color='black', linewidth=1)
    for y in range(grid_size[1] + 1):
        ax.plot([y, y], [0, grid_size[0]], color='black', linewidth=1)

    # Add state values
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            state = x * grid_size[1] + y
            value = values[state]
            action = policy[state]
            ax.text(y + 0.5, grid_size[0] - x - 0.5, f'{value:.2f}', 
                    color='blue', ha='center', va='center', fontsize=10)
            
            # Add policy arrows
            if action == 0:  # North
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, 0, 0.3, head_width=0.1, color='green')
            elif action == 1:  # Northeast
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, 0.2, 0.2, head_width=0.1, color='green')
            elif action == 2:  # East
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, 0.3, 0, head_width=0.1, color='green')
            elif action == 3:  # Southeast
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, 0.2, -0.2, head_width=0.1, color='green')
            elif action == 4:  # South
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, 0, -0.3, head_width=0.1, color='green')
            elif action == 5:  # Southwest
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, -0.2, -0.2, head_width=0.1, color='green')
            elif action == 6:  # West
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, -0.3, 0, head_width=0.1, color='green')
            elif action == 7:  # Northwest
                ax.arrow(y + 0.5, grid_size[0] - x - 0.5, -0.2, 0.2, head_width=0.1, color='green')

    # Set limits and labels
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Environment Visualization')
    plt.gca().invert_yaxis()
    plt.show()
