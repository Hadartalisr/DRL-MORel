from stable_baselines3 import DDPG
from stable_baselines3 import SAC

from DataUtils import DataUtils
from EnvUtils import EnvUtils

from PRIMORLPolicy import PRIMORLPolicy
# env = EnvUtils.get_human_rendering_env()
#
# # base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
# # policy = DDPG("MlpPolicy", env,  verbose=0)
#
# base_policies_dir_name = DataUtils.get_PRIMORL_agent_data_dir_name()
# policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name)
# policy = SAC.load(policy_filepath)
#
# vec_env = policy.get_env()
# obs = vec_env.reset()
# while True:
#     action, _states = policy.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     if dones[0]:
#         break
#     env.render()





# Load the environment for rendering
env = EnvUtils.get_human_rendering_env()

# Load a saved SAC policy
base_policies_dir_name = DataUtils.get_PRIMORL_agent_data_dir_name()
policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name)
agent = PRIMORLPolicy(env)
agent.model.load(policy_filepath)

vec_env = agent.model.get_env()
obs = vec_env.reset()

# Run the policy in the environment
while True:
    print(obs.shape)
    action, _states = agent.model.predict(obs)
    obs, rewards, dones, info, _ = env.step(action)
    if dones:
        break
    obs = obs.T
    env.render()
