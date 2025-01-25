import numpy as np
import matplotlib.pyplot as plt

import BasePolicyTrajectoriesGeneration
import Constants
from DataUtils import DataUtils
from EnvEnsembled import EnvEnsembled
from MORelDynamics import MORelDynamics

import numpy as np
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch

from PRIMORelDynamics import PRIMORelDynamics


def calculate_state_reward_error(env, states, actions, rewards):
    """
    Calculate the error in state and reward between the given trajectory and the model.

    Args:
        env (EnvEnsembled): The environment wrapper using the dynamics model.
        states (np.ndarray): The states from the trajectory.
        actions (np.ndarray): The actions from the trajectory.
        rewards (np.ndarray): The rewards from the trajectory.

    Returns:
        dict: A dictionary containing state error (MSE), reward error (MSE), and per-step errors.
    """
    predicted_states = []
    predicted_rewards = []

    # Initialize the model environment
    env.reset()
    current_state = torch.tensor(states[0], dtype=torch.float32)  # Convert to tensor
    env.state = current_state

    for i, action in enumerate(actions):
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Use the environment to predict the next state and reward
        next_state, reward, _, _, _ = env.step(action_tensor)

        # Store predictions
        predicted_states.append(next_state)
        predicted_rewards.append(reward)

        # Update current state to the actual next state from the trajectory
        env.state = torch.tensor(states[i + 1], dtype=torch.float32) if i + 1 < len(states) else None

    # Convert to numpy arrays
    predicted_states = np.array(predicted_states)
    predicted_rewards = np.array(predicted_rewards)

    # Calculate state and reward errors
    state_errors = np.linalg.norm(states - predicted_states, axis=1)
    reward_errors = np.abs(rewards - predicted_rewards)

    state_mse = np.mean(state_errors ** 2)
    reward_mse = np.mean(reward_errors ** 2)

    # Plot errors and predictions vs actual
    plt.figure(figsize=(16, 12))

    # State Predictions vs Actual
    plt.subplot(3, 1, 1)
    for dim in range(states.shape[1]):
        plt.plot(states[1:, dim], label=f"Actual State Dim {dim+1}", linestyle="--")
        plt.plot(predicted_states[:, dim], label=f"Predicted State Dim {dim+1}")
    plt.title("State Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("State Values")
    plt.legend()
    plt.grid()

    # State Errors
    plt.subplot(3, 1, 2)
    plt.plot(state_errors, label="State Error", color="blue")
    plt.title("State Prediction Errors")
    plt.xlabel("Time Step")
    plt.ylabel("Error (L2 Norm)")
    plt.legend()
    plt.grid()

    # Reward Predictions vs Actual
    plt.subplot(3, 1, 3)
    plt.plot(rewards, label="Actual Rewards", linestyle="--", color="orange")
    plt.plot(predicted_rewards, label="Predicted Rewards", color="green")
    plt.title("Reward Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Reward Values")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return {
        "state_mse": state_mse,
        "reward_mse": reward_mse,
        "state_errors": state_errors,
        "reward_errors": reward_errors,
    }

# Example usage in the main function
def main():
    trajectories_data_dir_name = DataUtils.get_trajectories_data_dir_name()
    file_path = DataUtils.get_random_file_path(trajectories_data_dir_name)
    states, actions, rewards = DataUtils.load_trajectory(file_path)
    # BasePolicyTrajectoriesGeneration.plot_trajectory(states, actions, rewards, should_save=False)

    input_dim = 4
    output_dim = 4
    model_name = "68092147-eb06-4f6f-9758-ef010a845adf"
    # dynamics_ensemble = MORelDynamics(input_dim=input_dim,  # TODO unite with PRIMORALDPPMAIN
    #                                    output_dim=output_dim,
    #                                    n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
    #                                    n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
    #                                    name=model_name)
    #
    # dynamics_ensemble.load()
    dynamics_ensemble = PRIMORelDynamics(input_dim=input_dim,  # TODO unite with PRIMORALDPPMAIN
                                         output_dim=output_dim,
                                         n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
                                         n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
                                         clipping_norm=Constants.PRIMORL_CLIPPING_NORM,
                                         noise_multiplier=Constants.PRIMORL_NOISE_MULTIPLIER_HIGH,
                                         sampling_ratio=Constants.PRIMORL_SAMPLING_RATIO,
                                         name=model_name)

    dynamics_ensemble.load()

    env = EnvEnsembled(dynamics_model=dynamics_ensemble,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       timeout_steps=Constants.PRIMORL_MODEL_MAX_TIME_STEPS,
                       uncertain_threshold=Constants.PRIMORL_UNCERTAINTY_THRESHOLD,
                       uncertain_penalty=Constants.PRIMORL_UNCERTAINTY_PENALTY)

    errors = calculate_state_reward_error(env, states, actions, rewards)

    print("State MSE:", errors["state_mse"])
    print("Reward MSE:", errors["reward_mse"])

if __name__ == '__main__':
    main()

