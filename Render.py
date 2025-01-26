import os


import Constants
from DataUtils import DataUtils
from EnvUtils import EnvUtils
from BasePolicy import BasePolicy
from MORelPolicy import MORelPolicy


def render_base_policy():
    env = EnvUtils.get_human_rendering_env()
    base_policy = BasePolicy(env)
    vec_env = base_policy.nn.get_env()

    base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
    base_policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name, "22000")
    print(base_policy_filepath)
    base_policy.load_model(base_policy_filepath)

    obs = vec_env.reset()

    while True:
        action, _states = base_policy.nn.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            break
        env.render()



def render_agent_policy():
    env = EnvUtils.get_human_rendering_env()
    agent = MORelPolicy(env)
    vec_env = agent.nn.get_env()

    dir_name = DataUtils.get_MORel_agent_data_dir_name()
    policy_filepath = DataUtils.get_random_file_path(dir_name)
    print(policy_filepath)
    agent.load_model(policy_filepath)

    obs = vec_env.reset()

    while True:
        action, _states = agent.nn.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            break
        env.render()


import os
import gymnasium
from gymnasium.wrappers import RecordVideo

def render_base_policy_to_video(output_folder="./save_videos3", max_steps=1000, episodes=3):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Create the environment and validate it
    try:
        env = gymnasium.make(Constants.ENV_NAME, render_mode="rgb_array")
        if not isinstance(env, gymnasium.Env):
            raise TypeError(f"The environment {Constants.ENV_NAME} is not a valid gymnasium.Env.")
        print(f"Successfully created environment: {Constants.ENV_NAME}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # Wrap the environment with RecordVideo
    try:
        env = RecordVideo(env, video_folder=output_folder, video_length=max_steps, disable_logger=True)
        print(f"Recording videos to: {output_folder}")
    except Exception as e:
        print(f"Error wrapping environment with RecordVideo: {e}")
        env.close()
        return

    # Initialize the base policy
    base_policy = BasePolicy(env)
    vec_env = base_policy.nn.get_env()

    # Load the base policy model
    base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
    base_policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name, "22000")
    print(f"Loading base policy model from: {base_policy_filepath}")
    base_policy.load_model(base_policy_filepath)

    # Iterate through episodes
    for episode in range(episodes):
        print(f"Starting episode {episode + 1}")
        obs = vec_env.reset()
        step_count = 0  # Track steps for the episode

        while step_count < max_steps:
            try:
                # Predict and perform the action
                action, _states = base_policy.nn.predict(obs)
                obs, rewards, dones, info = vec_env.step(action)

                # Render environment and count steps
                env.render()
                step_count += 1

                # Stop if the episode is done
                if dones[0]:
                    print(f"Episode {episode + 1} finished after {step_count} steps.")
                    break
            except Exception as e:
                print(f"Error during simulation: {e}")
                break

    # Close the environment
    env.close()

    # Count saved videos
    video_count = len([file for file in os.listdir(output_folder) if file.endswith(".mp4")])
    print(f"Total videos saved: {video_count}")




# Call the function



if __name__ == "__main__":
    # render_base_policy()
    #  render_agent_policy()
    render_base_policy_to_video("Figures/Pendulum-v1/Gifs/Videos")

