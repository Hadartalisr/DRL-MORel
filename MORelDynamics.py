import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from DataUtils import DataUtils


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


class MORelDynamics:
    def __init__(self, input_dim, output_dim, n_models, n_neurons, name, activation=nn.ReLU):
        self.n_models = n_models
        self.models = [DynamicsNet(input_dim, output_dim, n_neurons, activation) for _ in range(n_models)]
        self.name = name

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

    def save(self):
        dir = self.get_model_data_dir()
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(dir, f"{i}.pth"))

    def load(self):
        dir = self.get_model_data_dir()
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(dir, f"{i}.pth"), weights_only=True))

    def predict(self, x):
        with torch.no_grad():
            return torch.stack([model(x) for model in self.models])


    def get_model_data_dir(self):
        dir = DataUtils.get_MORel_model_data_dir_name()
        model_dir = os.path.join(dir, self.name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir