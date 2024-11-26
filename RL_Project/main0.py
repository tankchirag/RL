from environment import Environment
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration

def main():
    env = Environment(grid_size=(4, 4), goal_state=(3, 3), hell_state=(2, 2))
    env.initialize_environment()

    print("Running Value Iteration...")
    vi = ValueIteration(env)
    vi_values, vi_policy = vi.run()
    print("Optimal Values (Value Iteration):", vi_values)
    print("Optimal Policy (Value Iteration):", vi_policy)

    print("\nRunning Policy Iteration...")
    pi = PolicyIteration(env)
    pi_values, pi_policy = pi.run()
    print("Optimal Values (Policy Iteration):", pi_values)
    print("Optimal Policy (Policy Iteration):", pi_policy)

    # #visualization
    # visualize_policy(env, vi.policy, vi.values)

if __name__ == "__main__":
    main()
