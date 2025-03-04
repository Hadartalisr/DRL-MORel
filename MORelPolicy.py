import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, DDPG
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


class MORelPolicy:
    def __init__(self, env):
        self.env = env
        self.logger = TrainingLogger()
        self.nn = DDPG("MlpPolicy", self.env, verbose=1, learning_rate=1e-4)

    def train(self, total_timesteps):
        self.nn.learn(total_timesteps=total_timesteps,
                         log_interval=100,
                         callback=self.logger)

    def save(self):
        # Save the model
        base_policy_filepath, base_policy_name = DataUtils.get_new_MORLel_agent_filepath()
        self.nn.save(base_policy_filepath)

        # Extract and plot data
        timesteps_rewards, rewards = zip(*self.logger.rewards)

        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps_rewards, rewards, label="Rewards", marker='o')
        plt.title(f"Rewards During Training (Policy {base_policy_name})")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

        MORel_policies_figs_dir_name = DataUtils.get_MORel_agent_figs_dir_name()
        plt.savefig(MORel_policies_figs_dir_name + f"/{base_policy_name}.png")
        plt.close()

    def close_env(self):
        """Close the environment."""
        self.env.close()


    def load_model(self, model_path):
        self.nn = DDPG.load(model_path)

