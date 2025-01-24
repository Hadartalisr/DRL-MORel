import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

class EnvEnsembled(gym.Env):
    def __init__(self, dynamics_model, input_dim, output_dim, timeout_steps, uncertain_threshold, uncertain_penalty=-100):
        super(EnvEnsembled, self).__init__()
        self.dynamics_model = dynamics_model
        self.uncertain_penalty = uncertain_penalty
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.timeout_steps = timeout_steps
        self.uncertain_threshold = uncertain_threshold

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -8.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 8.0], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.steps_elapsed = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment using a random state sampled from the dynamics model.
        """
        super().reset(seed=seed)
        self.state = torch.normal(mean=0, std=1, size=(self.input_dim - 1,)).float()
        self.steps_elapsed = 0
        return self.state.numpy(), {}

    def step(self, action):
        """
        Perform a step in the environment using the given action.
        """
        action = torch.tensor(action, dtype=torch.float32)

        # Predict next state and reward using all models in the dynamics model ensemble
        state_action = torch.cat([self.state, action], dim=-1)  # Ensure dimensions align correctly
        predictions = self.dynamics_model.predict(state_action)

        deltas = predictions[:, 0:self.output_dim - 1]
        rewards = predictions[:, -1]

        # Calculate next state
        deltas_mean = torch.mean(deltas, dim=0)
        next_obs = self.state + deltas_mean
        self.state = next_obs

        # Check for uncertainty
        uncertain = self.usad(predictions)
        # uncertain = False

        reward_out = torch.mean(rewards).item()

        if uncertain:
            reward_out = self.uncertain_penalty

        self.steps_elapsed += 1

        done = uncertain or self.steps_elapsed > self.timeout_steps

        return next_obs.numpy(), reward_out, done, {"HALT": uncertain}, {}

    def usad(self, predictions):
        """
        Compute uncertainty based on the predictions.
        For example, check if the predictions' variance exceeds a threshold.
        """
        distances = torch.std(predictions, dim=0)
        return sum(distances) > self.uncertain_threshold

    def render(self, mode="human"):
        """
        Render the environment (optional, can be customized).
        """
        pass

    def close(self):
        """
        Clean up resources when closing the environment.
        """
        pass
