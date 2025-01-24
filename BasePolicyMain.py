
from EnvUtils import EnvUtils
from BasePolicyTrainer import BasePolicyTrainer

def main():
    env = EnvUtils.get_env()
    number_of_policies = 1
    for i in range(number_of_policies):
        total_time_steps = 1000
        BasePolicyTrainer.train_and_save_policy(env,  total_time_steps)
    env.close()


if __name__ == "__main__":
    main()