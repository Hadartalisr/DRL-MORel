import BasePolicyTrajectoriesGeneration
from EnvUtils import EnvUtils
from BasePolicy import BasePolicy
from BasePolicyTrainer import BasePolicyTrainer


def main():
    env = EnvUtils.get_env()
    policy = BasePolicy(env)
    number_of_iterations = 100
    total_time_steps_per_iteration = 200
    total_time_steps = 0
    for i in range(number_of_iterations):
        env.reset()
        total_time_steps += total_time_steps_per_iteration
        base_policy_name = BasePolicyTrainer.train_and_save_policy(policy=policy,
                                                training_time_steps=total_time_steps_per_iteration,
                                                total_time_steps=total_time_steps)
        BasePolicyTrajectoriesGeneration.generate_trajectory(env, policy, base_policy_name)
    env.close()




if __name__ == "__main__":
    main()

