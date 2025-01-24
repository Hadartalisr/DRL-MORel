from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np

import Constants
from DataUtils import DataUtils
from PRIMORLDynamics import PRIMORLDynamics
from EnsembledEnv import EnsembledEnv
from PRIMORLPolicy import PRIMORLPolicy
from TDPModelTraining import TDPModelTraining  # NEW IMPORT


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_files):
        self.trajectory_files = trajectory_files

    def __len__(self):
        return len(self.trajectory_files)

    def __getitem__(self, idx):
        file_path = self.trajectory_files[idx]
        sars_array = DataUtils.load_trajectory_as_sars_array(file_path)
        states, actions, rewards, next_states = zip(*sars_array)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)

        inputs = np.hstack((states, actions))
        targets = np.hstack((next_states, rewards))

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


# Main training function
def main():
    summary_writer = SummaryWriter(log_dir="tensorboard_logs")

    # Create dataset and dataloader
    trajectory_files = DataUtils.get_files_paths(DataUtils.get_trajectories_data_dir_name())
    dataset = TrajectoryDataset(trajectory_files)
    dataloader = DataLoader(dataset,
                            batch_size=Constants.PRIMORL_BATCH_SIZE,
                            shuffle=True)

    # Initialize dynamics ensemble
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[1]
    output_dim = dataset[0][1].shape[1]
    # Prepare TDP model training
    tdp_trainer = TDPModelTraining(input_dim=input_dim,
                                   output_dim=output_dim,
                                   n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
                                   n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
                                   clipping_norm=Constants.PRIMORL_CLIPPING_NORM,
                                   noise_multiplier=Constants.PRIMORL_NOISE_MULTIPLIER_HIGH,
                                   sampling_ratio=Constants.PRIMORL_SAMPLING_RATIO
                                   )

    # Train dynamics ensemble with TDP (Epoch-based training reintroduced)
    tdp_trainer.train(dataloader=dataloader,
                      epochs=Constants.PRIMORL_MODEL_LEARNING_NUMBER_OF_EPOCHS,
                      batch_size=Constants.PRIMORL_BATCH_SIZE
                      )

    env = EnsembledEnv(dynamics_model=tdp_trainer,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       timeout_steps=Constants.PRIMORL_MODEL_MAX_TIME_STEPS,
                       uncertain_threshold=Constants.PRIMORL_UNCERTAINTY_THRESHOLD,
                       uncertain_penalty=Constants.PRIMORL_UNCERTAINTY_PENALTY)

    agent = PRIMORLPolicy(env)
    agent.train(total_timesteps=Constants.PRIMORL_AGENT_LEARNING_TOTAL_TIME_STEPS)
    agent.save()


if __name__ == "__main__":
    main()
