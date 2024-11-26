from environment import GridEnvironment
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from generalized_policy_iteration import GeneralizedPolicyIteration
from visualization import plot_policy, plot_convergence

if __name__ == "__main__":
    # Initialize the environment
    rows, cols = 4, 4
    terminal_states = [(0, 2), (2, 1)]
    rewards = {(0, 2): 1.0, (2, 1): -1.0}
    env = GridEnvironment(rows, cols, terminal_states, rewards)

    # Run Value Iteration
    vi = ValueIteration(env)
    track_vi = []
    policy_vi, values_vi, iterations_vi = vi.run_value_iteration(track_convergence=track_vi)

    # Run Policy Iteration
    pi = PolicyIteration(env)
    policy_pi, values_pi, iterations_pi = pi.run_policy_iteration()

    # Run Generalized Policy Iteration
    gpi = GeneralizedPolicyIteration(env)
    policy_gpi, values_gpi, iterations_gpi = gpi.run_gpi()

    # Print results
    print(f"Value Iteration: {iterations_vi} iterations")
    print(f"Policy Iteration: {iterations_pi} iterations")
    print(f"Generalized Policy Iteration: {iterations_gpi} iterations")

    # Visualization
    plot_policy(policy_vi, rows, cols, title="Value Iteration Policy")
    plot_policy(policy_pi, rows, cols, title="Policy Iteration Policy")
    plot_policy(policy_gpi, rows, cols, title="Generalized Policy Iteration Policy")
    plot_convergence(track_vi, "Value Iteration")
