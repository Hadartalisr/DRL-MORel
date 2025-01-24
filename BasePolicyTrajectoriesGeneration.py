import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG

######################################################

from EnvUtils import EnvUtils
from DataUtils import DataUtils
from BasePolicy import BasePolicy

######################################################

def generate_trajectories(env, policy, number_of_trajectories):
    """
    Generate and save a specified number of trajectories without rendering the environment.
    """
    for i in range(number_of_trajectories):
        print(f"Generating trajectory {i + 1}/{number_of_trajectories}")
        states, actions, rewards = generate_trajectory(env, policy)
        DataUtils.save_trajectory(states, actions, rewards)

def generate_trajectory(env, policy, should_plot=False, policy_name=""):
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


def plot_trajectory(states, actions, rewards, policy_name="", should_save=True):

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
    if should_save:
        plt.savefig(base_policies_figs_dir_name + f"/{policy_name}.png")
    else:
        plt.show()
    plt.close()


def plot_trajectory_from_file_name(file_name):
    trajectories_data_dir_name = DataUtils.get_trajectories_data_dir_name()
    trajectory_file_name = f"{trajectories_data_dir_name}/{file_name}"
    states, actions, rewards = DataUtils.load_trajectory(trajectory_file_name)
    plot_trajectory(states, actions, rewards, should_save=False)



# Main code
if __name__ == "__main__":
    # plot trajectory
    # plot_trajectory_from_file_name("ff583718-d6a5-478c-8f3a-859e7218906f.npz")

    env = EnvUtils.get_env()
    policy = BasePolicy(env)
    policies_dir_name = DataUtils.get_base_policies_data_dir_name()
    policies_file_paths = DataUtils.get_files_paths(policies_dir_name)
    for policy_file_path in policies_file_paths:
        policy.load_model(policy_file_path)
        print(f"Loading policy from {policy_file_path}")
        generate_trajectories(env, policy, number_of_trajectories=10)
    env.close()
