import numpy as np
from stable_baselines3 import DDPG

######################################################

from EnvUtils import EnvUtils
from DataUtils import DataUtils

######################################################

def generate_trajectories(env, policy, number_of_trajectories):
    """
    Generate and save a specified number of trajectories without rendering the environment.
    """
    for i in range(number_of_trajectories):
        print(f"Generating trajectory {i + 1}/{number_of_trajectories}")
        states, actions, rewards = generate_trajectory(env, policy)
        DataUtils.save_trajectory(states, actions, rewards)

def generate_trajectory(env, policy):
    """
    Generate a single trajectory and return states, actions, and rewards.
    """
    states = []
    actions = []
    rewards = []

    state, _ = env.reset()
    while True:
        action, _ = policy.predict(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    return np.array(states), np.array(actions), np.array(rewards)

# Main code
if __name__ == "__main__":
    env = EnvUtils.get_env()

    for i in range(100):
        policy_dir_name = DataUtils.get_base_policies_data_dir_name()
        policy_filepath = DataUtils.get_random_file_path(policy_dir_name)
        print(f"Loading policy from {policy_filepath}")

        policy_network = DDPG.load(policy_filepath, env=env)
        generate_trajectories(env, policy_network, number_of_trajectories=10)

    env.close()
