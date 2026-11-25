import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, dim_layers):
        super(MLP, self).__init__()

        # define layers
        layers = []
        for d in range(len(dim_layers) - 1):
            layers.append(nn.Linear(dim_layers[d], dim_layers[d+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        X = X.reshape(X.shape[0], -1)   # [B, 3, 64, 64] -> [B, 12288]

        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i < len(self.layers) - 1:
                X = F.relu(X)

        return F.log_softmax(X, dim=1)