import os

# Define the folder structure
folders = [
    "RL_Project/src",
    "RL_Project/config",
    "RL_Project/data/stats",
    "RL_Project/data/visualizations",
    "RL_Project/tests",
]

files = [
    "RL_Project/src/__init__.py",
    "RL_Project/src/grid_environment.py",
    "RL_Project/src/agent.py",
    "RL_Project/src/policy_iteration.py",
    "RL_Project/src/value_iteration.py",
    "RL_Project/src/generalization.py",
    "RL_Project/src/visualizer.py",
    "RL_Project/src/utils.py",
    "RL_Project/main.py",
    "RL_Project/config/settings.json",
    "RL_Project/README.md",
    "RL_Project/tests/test_agent.py",
    "RL_Project/tests/test_environment.py",
    "RL_Project/tests/test_policy_iteration.py",
    "RL_Project/tests/test_value_iteration.py",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    with open(file, "w") as f:
        # Add a basic structure to Python files
        if file.endswith(".py"):
            f.write('"""\nAuto-generated file\n"""\n\n')
        if file.endswith(".json"):
            f.write("{}\n")  # Empty JSON object
        if "README.md" in file:
            f.write("# RL Project\n\nProject description here.")

print("Folder structure and files created successfully.")
