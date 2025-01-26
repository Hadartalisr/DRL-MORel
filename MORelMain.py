from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np

import BasePolicyTrajectoriesGeneration
import Constants
from DataUtils import DataUtils
from MORelDynamics import MORelDynamics
from EnvEnsembled import EnvEnsembled
from MORelPolicy import MORelPolicy
from PRIMORelDynamics import PRIMORelDynamics


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
def dynamics_learning_main():
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
    dynamics_name = DataUtils.generate_unique_string_uuid()

    # dynamics_ensemble = MORelDynamics(input_dim=input_dim,  #TODO unite with PRIMORALDPPMAIN
    #                                   output_dim=output_dim,
    #                                   n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
    #                                   n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
    #                                   name=dynamics_name)

    dynamics_ensemble = PRIMORelDynamics(input_dim=input_dim,
                                         output_dim=output_dim,
                                         n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
                                         n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
                                         clipping_norm=Constants.PRIMORL_CLIPPING_NORM,
                                         noise_multiplier=Constants.PRIMORL_NOISE_MULTIPLIER_HIGH,
                                         sampling_ratio=Constants.PRIMORL_SAMPLING_RATIO,
                                         name=dynamics_name
                                         )

    # Prepare optimizers and loss functions
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in dynamics_ensemble.models]
    loss_fns = [nn.MSELoss() for _ in range(dynamics_ensemble.n_models)]

    # Train dynamics ensemble
    dynamics_ensemble.train(
        dataloader=dataloader,
        epochs=Constants.PRIMORL_MODEL_LEARNING_NUMBER_OF_EPOCHS,
        optimizers=optimizers,
        summary_writer=summary_writer
    )

    dynamics_ensemble.save()





def policy_learning_main():
    trajectory_files = DataUtils.get_files_paths(DataUtils.get_trajectories_data_dir_name())
    dataset = TrajectoryDataset(trajectory_files)
    dataloader = DataLoader(dataset,
                            batch_size=Constants.PRIMORL_BATCH_SIZE,
                            shuffle=True)

    # Initialize dynamics ensemble
    sample_input, _ = dataset[0]
    # this is already with the action as the last indexc922b825-cdac-4a6e-95a4-04a5d9c3b326
    input_dim = sample_input.shape[1]
    # this is already with the reward as the last index
    output_dim = dataset[0][1].shape[1]

    model_name = "68092147-eb06-4f6f-9758-ef010a845adf"

    # dynamics_ensemble = MORelDynamics(input_dim=input_dim,  # TODO unite with PRIMORALDPPMAIN
    #                                   output_dim=output_dim,
    #                                   n_models=Constants.PRIMORL_ENSEMBLE_SIZE,
    #                                   n_neurons=Constants.PRIMORL_MODEL_NEURONS_PER_LAYER,
    #                                   name=model_name)
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
                       uncertain_threshold= Constants.PRIMORL_UNCERTAINTY_THRESHOLD,
                       uncertain_penalty=Constants.PRIMORL_UNCERTAINTY_PENALTY)


    agent = MORelPolicy(env)
    agent.train(total_timesteps=Constants.PRIMORL_AGENT_LEARNING_TOTAL_TIME_STEPS)
    BasePolicyTrajectoriesGeneration.generate_trajectory(env, agent,
                                                         should_plot=True,
                                                         should_save=False,
                                                         policy_name=model_name)
    agent.save()


if __name__ == "__main__":
    dynamics_learning_main()
    # policy_learning_main()


