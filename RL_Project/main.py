from environment import GridEnvironment
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from generalized_policy_iteration import GeneralizedPolicyIteration
from visualization import GridVisualizer

def compare_methods(env):
    """
    Compare Value Iteration, Policy Iteration, and Generalized Policy Iteration.

    :param env: An instance of GridEnvironment.
    """
    vi = ValueIteration(env)
    pi = PolicyIteration(env)
    gpi = GeneralizedPolicyIteration(env)

    policy_vi, values_vi, iterations_vi = vi.run()
    policy_pi, values_pi, iterations_pi = pi.run()
    policy_gpi, values_gpi, iterations_gpi = gpi.iterate()

    print("Value Iteration:", iterations_vi, "iterations")
    print("Policy Iteration:", iterations_pi, "iterations")
    print("Generalized Policy Iteration:", iterations_gpi, "iterations")

    visualizer = GridVisualizer(env)
    visualizer.display_grid(values_vi, policy_vi)

if __name__ == "__main__":
    rows, cols = 10, 10
    terminal_states = [(3, 7), (6, 1)]
    rewards = {(0, 2): 1, (2, 1): -1}

    env = GridEnvironment(rows, cols, terminal_states, rewards)
    compare_methods(env)
