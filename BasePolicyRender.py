from stable_baselines3 import DDPG

from DataUtils import DataUtils
from EnvUtils import EnvUtils


env = EnvUtils.get_human_rendering_env()
base_policy = DDPG("MlpPolicy", env,  verbose=0)
vec_env = base_policy.get_env()

base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
base_policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name)
base_policy = DDPG.load(base_policy_filepath)

obs = vec_env.reset()
while True:
    action, _states = base_policy.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    if dones[0]:
        break
    env.render()


