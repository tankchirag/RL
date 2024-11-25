import matplotlib.pyplot as plt
import numpy as np

class GridVisualizer:
    """
    Visualizes the grid environment and agent policies.
    """

    def __init__(self, env):
        """
        Initialize the visualizer.

        :param env: An instance of GridEnvironment.
        """
        self.env = env

    def display_grid(self, values=None, policy=None):
        """
        Display the grid with values and policy.

        :param values: Dictionary of state values.
        :param policy: Dictionary of state policies.
        """
        grid = self.env.grid
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap="coolwarm", interpolation="nearest")

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                value = values.get((row, col), 0) if values else 0
                action = policy.get((row, col), None) if policy else None
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="black")

                if action:
                    dx, dy = action
                    ax.arrow(
                        col, row, 0.3 * dy, -0.3 * dx,
                        head_width=0.1, head_length=0.2, fc="black", ec="black"
                    )

        plt.show()
