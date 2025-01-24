import random

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from DataUtils import DataUtils
from EnvUtils import EnvUtils
import Constants

from BasePolicy import BasePolicy


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


class BasePolicyTrainer:

    @staticmethod
    def train_and_save_policy(env, training_time_steps):
        """
        Train a single policy and save its model and reward plot.
        """

        # Create and train the model
        policy = BasePolicy(env)
        training_logger = TrainingLogger()
        policy.nn.learn(total_timesteps=training_time_steps,
                          log_interval=10,
                          callback=training_logger)

        # Save the model
        base_policy_filepath, base_policy_name = DataUtils.get_new_base_policy_filepath(training_time_steps)
        policy.nn.save(base_policy_filepath)

        # Extract and plot data
        time_steps_rewards, rewards = zip(*training_logger.rewards)

        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps_rewards, rewards, label="Rewards", marker='o')
        plt.title(f"Rewards During Training (Policy {base_policy_name})")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

        base_policies_figs_dir_name = DataUtils.get_base_policies_figs_dir_name()
        plt.savefig(base_policies_figs_dir_name + f"/{base_policy_name}.png")
        plt.close()


