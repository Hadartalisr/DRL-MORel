import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from EnsembleClipping import ensemble_clip


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


class TDPModelTraining:
    def __init__(self, input_dim, output_dim, n_models, n_neurons,
                 clipping_norm, noise_multiplier, sampling_ratio, activation=nn.ReLU):
        self.models = [DynamicsNet(input_dim, output_dim, n_neurons, activation) for _ in range(n_models)]
        self.clipping_norm = clipping_norm
        self.noise_multiplier = noise_multiplier
        self.sampling_ratio = sampling_ratio

    def forward(self, model_idx, x):
        return self.models[model_idx](x)

    def train(self, dataloader, epochs, batch_size, summary_writer=None):
        # Initialize optimizers for each model in the ensemble
        optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in self.models]
        # Define the loss function (Mean Squared Error)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(epochs):
            for batch_idx, (feed, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
                gradients = []
                losses = []

                # Compute gradients for each model
                for model, optimizer in zip(self.models, optimizers):
                    optimizer.zero_grad()
                    predictions = model(feed)
                    loss = loss_fn(predictions, target)
                    loss.backward()
                    losses.append(loss.item())

                    # Collect the gradients of the current model
                    gradients.append([param.grad.clone() for param in model.parameters()])

                # Apply ensemble gradient clipping
                clipped_gradients = ensemble_clip(gradients, self.clipping_norm)

                # Add noise to gradients and update model parameters
                for model, optimizer, grads in zip(self.models, optimizers, clipped_gradients):
                    for param, grad in zip(model.parameters(), grads):
                        noise = torch.normal(mean=0, std=self.noise_multiplier, size=grad.shape)
                        param.grad = grad + noise
                    optimizer.step()

                # Log to TensorBoard
                if summary_writer:
                    avg_loss = sum(losses) / len(losses)
                    for i, loss_val in enumerate(losses):
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
