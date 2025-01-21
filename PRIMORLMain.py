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
    # this is already with the action as the last index
    input_dim = sample_input.shape[1]
    # this is already with the reward as the last index
    output_dim = dataset[0][1].shape[1]
    dynamics_ensemble = PRIMORLDynamics(input_dim=input_dim,
                                        output_dim=output_dim,
                                        n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
                                        n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER)

    # Prepare optimizers and loss functions
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in dynamics_ensemble.models]
    loss_fns = [nn.MSELoss() for _ in range(dynamics_ensemble.n_models)]

    # Train dynamics ensemble
    dynamics_ensemble.train(
        dataloader=dataloader,
        epochs=Constants.PRIMORL_MODEL_NUMBER_OF_EPOCHS,
        optimizers=optimizers,
        loss_fns=loss_fns,
        summary_writer=summary_writer
    )

    env = EnsembledEnv(dynamics_model=dynamics_ensemble,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       timeout_steps=Constants.PRIMORL_MODEL_MAX_TIME_STEPS,
                       uncertain_penalty=Constants.PRIMORL_UNCERTAINTY_PENALTY)


    agent = PRIMORLPolicy(env)
    agent.train(total_timesteps=Constants.PRIMORL_AGENT_MAX_TIME_STEPS)





if __name__ == "__main__":
    main()