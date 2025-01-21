import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from DataUtils import DataUtils
import Constants


class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons, activation):
        super(DynamicsNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.h0 = nn.Linear(n_neurons, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.h0_act(self.h0(x))
        x = self.h1_act(self.h1(x))
        x = self.output_layer(x)
        return x


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

class DynamicsEnsemble():
    def __init__(self, input_dim, output_dim, n_models=4, n_neurons=512, threshold=1.5, activation=nn.ReLU):
        self.n_models = n_models
        self.threshold = threshold
        self.models = [DynamicsNet(input_dim, output_dim, n_neurons, activation) for _ in range(n_models)]

    def forward(self, model_idx, x):
        return self.models[model_idx](x)

    def train_step(self, model_idx, feed, target, optimizer, loss_fn):
        optimizer.zero_grad()
        next_state_pred = self.models[model_idx](feed)
        loss = loss_fn(next_state_pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, dataloader, epochs, optimizers, loss_fns, summary_writer):
        for epoch in range(epochs):
            for batch_idx, (feed, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                loss_vals = [
                    self.train_step(i, feed, target, optimizers[i], loss_fns[i])
                    for i in range(self.n_models)
                ]

                avg_loss = sum(loss_vals) / len(loss_vals)

                # Log to TensorBoard
                if summary_writer:
                    for i, loss_val in enumerate(loss_vals):
                        summary_writer.add_scalar(f'Loss/model_{i}', loss_val, epoch * len(dataloader) + batch_idx)
                    summary_writer.add_scalar('Loss/avg', avg_loss, epoch * len(dataloader) + batch_idx)

    def save(self, save_dir):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(save_dir, f"dynamics_{i}.pt"))

    def load(self, load_dir):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(load_dir, f"dynamics_{i}.pt")))

    def predict(self, x):
        with torch.no_grad():
            return torch.stack([model(x) for model in self.models])

# Main training function
def main():
    summary_writer = SummaryWriter(log_dir="tensorboard_logs")

    # Get trajectory files
    trajectory_files = DataUtils.get_files_paths(DataUtils.get_trajectories_data_dir_name())

    # Create dataset and dataloader
    dataset = TrajectoryDataset(trajectory_files)
    dataloader = DataLoader(dataset,
                            batch_size=Constants.MOREL_BATCH_SIZE,
                            shuffle=True)

    # Initialize dynamics ensemble
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[1]
    output_dim = dataset[0][1].shape[1]
    dynamics_ensemble = DynamicsEnsemble(input_dim,
                                         output_dim,
                                         n_models=Constants.MOREL_ENSEMBLE_SIZE)

    # Prepare optimizers and loss functions
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in dynamics_ensemble.models]
    loss_fns = [nn.MSELoss() for _ in range(dynamics_ensemble.n_models)]

    # Train dynamics ensemble
    dynamics_ensemble.train(
        dataloader=dataloader,
        epochs=10,
        optimizers=optimizers,
        loss_fns=loss_fns,
        summary_writer=summary_writer
    )

    # Save trained models
    save_dir = DataUtils.get_model_data_dir_name()
    dynamics_ensemble.save(save_dir)

    print(f"Models saved to {save_dir}")

if __name__ == "__main__":
    main()