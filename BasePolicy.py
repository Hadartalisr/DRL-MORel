import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from DataUtils import DataUtils
from EnvUtils import EnvUtils
import Constants


class BasePolicy:
    def __init__(self, env):
        _, action_space_dim = EnvUtils.get_env_dims(env)
        action_noise = NormalActionNoise(mean=np.zeros(action_space_dim),
                                         sigma=0.1 * np.ones(action_space_dim))
        self.nn = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)


