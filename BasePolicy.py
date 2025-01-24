import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from EnvUtils import EnvUtils


class BasePolicy:
    def __init__(self, env):
        _, action_space_dim = EnvUtils.get_env_dims(env)
        action_noise = NormalActionNoise(mean=np.zeros(action_space_dim),
                                         sigma=0.1 * np.ones(action_space_dim))
        self.nn = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=2)


