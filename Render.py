from DataUtils import DataUtils
from EnvUtils import EnvUtils
from BasePolicy import BasePolicy


def render_base_policy():
    env = EnvUtils.get_human_rendering_env()
    base_policy = BasePolicy(env)
    vec_env = base_policy.nn.get_env()

    base_policies_dir_name = DataUtils.get_base_policies_data_dir_name()
    base_policy_filepath = DataUtils.get_random_file_path(base_policies_dir_name, "1400")
    print(base_policy_filepath)
    base_policy.load_model(base_policy_filepath)

    obs = vec_env.reset()

    while True:
        action, _states = base_policy.nn.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            break
        env.render()


if __name__ == "__main__":
    render_base_policy()

