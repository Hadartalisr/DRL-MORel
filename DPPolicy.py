import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from DataUtils import DataUtils
from EnvUtils import EnvUtils
import Constants

class TrainingLogger(BaseCallback):
    """
    Custom callback to log rewards and losses during training.
    """
    def __init__(self, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Log rewards
        reward = np.mean(self.locals["rewards"])
        self.rewards.append((self.num_timesteps, reward))
        return True

def train_and_save_policy(env, policy_index):
    """
    Train a single policy and save its model and reward plot.
    """
    # Noise for DDPG
    _, action_space_dim = EnvUtils.get_env_dims(env)
    action_noise = NormalActionNoise(mean=np.zeros(action_space_dim), sigma=0.1 * np.ones(action_space_dim))

    # Create the logger callback
    training_logger = TrainingLogger()

    # Create and train the model
    base_policy = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    base_policy.learn(total_timesteps=Constants.BASE_POLICY_TRAIN_TOTAL_TIME_STEPS,
                      log_interval=10,
                      callback=training_logger)

    # Save the model
    base_policy_filepath, base_policy_name = DataUtils.get_new_base_policy_filepath()
    base_policy.save(base_policy_filepath)

    # Extract and plot data
    timesteps_rewards, rewards = zip(*training_logger.rewards)

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_rewards, rewards, label="Rewards", marker='o')
    plt.title(f"Rewards During Training (Policy {policy_index})")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    base_policies_figs_dir_name = DataUtils.get_base_policies_figs_dir_name()
    plt.savefig(base_policies_figs_dir_name + f"/{base_policy_name}.png")
    plt.close()

def main():
    env = EnvUtils.get_env()
    number_of_policies = 5
    for i in range(number_of_policies):
        print(f"Training policy {i + 1}/{number_of_policies}")
        train_and_save_policy(env, i + 1)
    env.close()

if __name__ == "__main__":
    main()
