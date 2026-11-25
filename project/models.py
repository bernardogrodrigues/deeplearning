import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, dim_layers, activation_fn=F.relu, output_fn=F.log_softmax, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        # define layers
        layers = []
        for d in range(len(dim_layers) - 2):
            layers.append(nn.Linear(dim_layers[d], dim_layers[d+1]))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        X = X.reshape(X.shape[0], -1)   # [B, 3, 64, 64] -> [B, 12288]

        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i < len(self.layers) - 1:
                X = self.activation_fn(X)

        return self.output_fn(X, dim=1)