import gymnasium as gym

######################################################

import Constants

######################################################


class EnvUtils:
    @staticmethod
    def get_env():
        env = gym.make(Constants.ENV_NAME)
        return env

    @staticmethod
    def get_human_rendering_env():
        env = gym.make(Constants.ENV_NAME, render_mode="human")
        return env

    @staticmethod
    def get_env_dims(env):
        state_space_dims = env.observation_space.shape[0]
        action_space_dims = env.action_space.shape[0]
        return state_space_dims, action_space_dims



