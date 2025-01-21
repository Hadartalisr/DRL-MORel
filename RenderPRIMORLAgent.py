from DataUtils import DataUtils
from EnvUtils import EnvUtils

from PRIMORLPolicy import PRIMORLPolicy

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
    action, _states = agent.model.predict(obs)
    obs, rewards, dones, info, _ = env.step(action)
    if dones:
        break
    obs = obs.T
    env.render()
