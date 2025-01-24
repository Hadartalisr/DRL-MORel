import gymnasium as gym
import Constants

class EnvBase(gym.Env):
    def __init__(self):
        super(EnvBase, self).__init__()
        self.env = gym.make(Constants.ENV_NAME)

    def __getattr__(self, name):
        """
        Forward any undefined attribute or method call to the underlying environment.
        """
        return getattr(self.env, name)

    def step(self, action):
        """
        Perform the step in the underlying environment and normalize the reward.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Normalize the reward
        normalized_reward = self._normalize_reward(reward)

        # Return the normalized reward along with other values
        return next_state, normalized_reward, terminated, truncated, info

    @staticmethod
    def _normalize_reward(reward):
        """
        Normalize the reward to a range of [0, 1].
        """
        return reward

    def reset(self, **kwargs):
        """
        Reset the underlying environment.
        """
        return self.env.reset(**kwargs)

