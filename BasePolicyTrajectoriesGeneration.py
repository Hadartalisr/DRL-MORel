import numpy as np
import matplotlib.pyplot as plt
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

def generate_trajectory(env, policy, policy_name, should_plot=False):
    """
    Generate a single trajectory and return states, actions, and rewards.
    """
    states = []
    actions = []
    rewards = []

    state, _ = env.reset()
    while True:
        action, _ = policy.nn.predict(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    if should_plot:
        plot_trajectory(states, actions, rewards, policy_name)
    return np.array(states), np.array(actions), np.array(rewards)


def plot_trajectory(states, actions, rewards, policy_name):

    """
    Plot states, actions, and rewards from a single trajectory.
    """

    time_steps = np.arange(len(rewards))
    base_policies_figs_dir_name = DataUtils.get_base_policies_figs_dir_name()

    # Unpack states (Pendulum-v1: cos(theta), sin(theta), angular velocity)
    cos_theta = states[:, 0]
    sin_theta = states[:, 1]
    angular_velocity = states[:, 2]

    # Plot angular position (cos(theta), sin(theta))
    plt.figure(figsize=(16, 6))
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, cos_theta, label="cos(theta)")
    plt.plot(time_steps, sin_theta, label="sin(theta)")
    plt.title("Pendulum Angular Position Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()

    # Plot angular velocity
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, angular_velocity, label="Angular Velocity", color="orange")
    plt.title("Pendulum Angular Velocity Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid()

    # Plot actions (torque)
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, actions, label="Torque", color="green")
    plt.title("Applied Torque Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Torque")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, rewards, label="Reward", color="red")
    plt.title("Rewards Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(base_policies_figs_dir_name + f"/{policy_name}.png")
    plt.close()



# Main code
if __name__ == "__main__":
    env = EnvUtils.get_env()

    for i in range(300):
        policy_dir_name = DataUtils.get_base_policies_data_dir_name()
        policy_filepath = DataUtils.get_random_file_path(policy_dir_name)
        print(f"Loading policy from {policy_filepath}")

        policy_network = DDPG.load(policy_filepath, env=env)
        generate_trajectories(env, policy_network, number_of_trajectories=10)

    env.close()
