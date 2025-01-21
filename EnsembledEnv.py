import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

class EnsembledEnv(gym.Env):
    def __init__(self, dynamics_model, input_dim, output_dim, timeout_steps=300, uncertain_penalty=-100):
        super(EnsembledEnv, self).__init__()
        self.dynamics_model = dynamics_model
        self.uncertain_penalty = uncertain_penalty
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.timeout_steps = timeout_steps

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.input_dim - 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim - 1,), dtype=np.float32)

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

        # Predict next state and reward using the dynamics model
        state_action = torch.cat([self.state, action], dim=-1)  # Ensure dimensions align correctly
        predictions = self.dynamics_model.predict(state_action.unsqueeze(0))

        deltas = predictions[:, 0:self.output_dim - 1]
        rewards = predictions[:, -1]

        # Calculate next state
        deltas_mean = torch.mean(deltas, dim=0)
        next_obs = self.state + deltas_mean
        self.state = next_obs

        # Check for uncertainty
        uncertain = self.dynamics_model.usad(predictions.numpy())

        reward_out = torch.mean(rewards).item()

        if uncertain:
            reward_out = self.uncertain_penalty

        self.steps_elapsed += 1

        done = uncertain or self.steps_elapsed > self.timeout_steps

        return next_obs.numpy(), reward_out, done, {"HALT": uncertain}

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
