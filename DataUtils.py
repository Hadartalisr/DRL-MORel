import os
import random
import numpy as np
import uuid


######################################################

import Constants

######################################################

######################################################
# Dir and Files Paths and Names
######################################################


class DataUtils:

    @staticmethod
    def create_dir_if_not_exists(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def get_env_data_dir_name():
        data_dir_name = f"{Constants.DATA_DIR_NAME}/{Constants.ENV_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(data_dir_name)
        return data_dir_name

    @staticmethod
    def get_env_figs_dir_name():
        figs_dir_name = f"{Constants.FIGS_DIR_NAME}/{Constants.ENV_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(figs_dir_name)
        return figs_dir_name

    @staticmethod
    def get_base_policies_data_dir_name():
        env_data_dir_name = DataUtils.get_env_data_dir_name()
        base_policies_dir_name = f"{env_data_dir_name}/{Constants.BASE_POLICIES_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(base_policies_dir_name)
        return base_policies_dir_name

    @staticmethod
    def get_PRIMORL_agent_data_dir_name():
        env_data_dir_name = DataUtils.get_env_data_dir_name()
        base_policies_dir_name = f"{env_data_dir_name}/{Constants.PRIMORL_AGENT_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(base_policies_dir_name)
        return base_policies_dir_name


    @staticmethod
    def get_base_policies_figs_dir_name():
        env_figs_dir_name = DataUtils.get_env_figs_dir_name()
        base_policies_figs_dir_name = f"{env_figs_dir_name}/{Constants.BASE_POLICIES_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(base_policies_figs_dir_name)
        return base_policies_figs_dir_name

    @staticmethod
    def get_PRIMORL_agent_figs_dir_name():
        env_figs_dir_name = DataUtils.get_env_figs_dir_name()
        base_policies_figs_dir_name = f"{env_figs_dir_name}/{Constants.PRIMORL_AGENT_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(base_policies_figs_dir_name)
        return base_policies_figs_dir_name

    @staticmethod
    def get_model_data_dir_name():
        env_data_dir_name = DataUtils.get_env_data_dir_name()
        model_data_dir_name = f"{env_data_dir_name}/{Constants.MODEL_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(model_data_dir_name)
        return model_data_dir_name

    @staticmethod
    def get_model_figs_dir_name():
        env_figs_dir_name = DataUtils.get_env_figs_dir_name()
        model_figs_dir_name = f"{env_figs_dir_name}/{Constants.MODEL_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(model_figs_dir_name)
        return model_figs_dir_name

    @staticmethod
    def get_trajectories_data_dir_name():
        env_data_dir_name = DataUtils.get_env_data_dir_name()
        trajectories_dir_name = f"{env_data_dir_name}/{Constants.TRAJECTORIES_DIR_NAME}"
        DataUtils.create_dir_if_not_exists(trajectories_dir_name)
        return trajectories_dir_name

    @staticmethod
    def get_new_trajectory_filepath():
        trajectories_dir_name = DataUtils.get_trajectories_data_dir_name()
        unique_str = DataUtils.generate_unique_string_uuid()
        return f"{trajectories_dir_name}/{unique_str}.npz"

    @staticmethod
    def get_new_base_policy_filepath():
        base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
        unique_str = DataUtils.generate_unique_string_uuid()
        return f"{base_policies_dir_name}/{unique_str}", unique_str

    @staticmethod
    def get_new_PRIMORL_agent_filepath():
        dir_name = DataUtils.get_PRIMORL_agent_data_dir_name()
        unique_str = DataUtils.generate_unique_string_uuid()
        return f"{dir_name}/{unique_str}", unique_str

    @staticmethod
    def generate_unique_string_uuid(prefix=""):
        unique_id = uuid.uuid4()
        return f"{prefix}{unique_id}"

    ######################################################

    ######################################################

    @staticmethod
    def get_files_paths(directory):
        """Returns a random file name from the given directory."""
        files = [f"{directory}/{file}" for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        return files

    @staticmethod
    def get_random_file_path(directory):
        """Returns a random file name from the given directory."""
        files = DataUtils.get_files_paths(directory)
        if not files:
            raise FileNotFoundError("No files found in the directory.")
        file_name = random.choice(files)
        return f"{file_name}"

    @staticmethod
    def save_trajectory(states, actions, rewards):
        new_trajectory_filepath = DataUtils.get_new_trajectory_filepath()
        np.savez(new_trajectory_filepath, states=states, actions=actions, rewards=rewards)

    @staticmethod
    def load_trajectory(file_path):
        data = np.load(file_path)
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        return states, actions, rewards

    @staticmethod
    def load_trajectory_as_sars_array(file_path):
        trajectory = DataUtils.load_trajectory(file_path)
        return DataUtils.trajectory_into_sars_array(trajectory)

    @staticmethod
    def trajectory_into_sars_array(trajectory):
        sars = []
        states, actions, rewards = trajectory
        actions = actions.reshape(-1, 1)  # Make actions 2D with shape (n, 1)
        for i in range(len(trajectory[0]) - 1):
            sars.append((states[i], actions[i], rewards[i], states[i + 1]))
        return sars







