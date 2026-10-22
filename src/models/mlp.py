# src/models/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import RearrangeableModel

class FC_MLP(RearrangeableModel):
    def __init__(self, in_dim: int, hidden_dims: list[int], num_classes: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        h = torch.flatten(x, 1)
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.fc_out(h)

    def get_layer_activations(self, x):
        activations = []
        h = torch.flatten(x, 1)
        for layer in self.layers:
            h = F.relu(layer(h))
            activations.append(h)
        return activations

    def forward_with_ablation(self, x, layer_idx, unit_idx):
        h = torch.flatten(x, 1)
        for idx, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if idx == layer_idx:
                h = h.clone()
                h[:, unit_idx] = 0.0
        return self.fc_out(h)

    def get_total_units(self):
        return sum(self.hidden_dims)

def build_3layer_mlp(in_dim=64, num_classes=10):
    return FC_MLP(in_dim, [128, 128, 64], num_classes)

def build_5layer_mlp(in_dim=64, num_classes=10):
    return FC_MLP(in_dim, [256, 256, 128, 128, 64], num_classes)
