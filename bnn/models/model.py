import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from .layers import Linear

class MLP(BNNModule):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(in_features=11, out_features=20)
        self.linear2 = Linear(in_features=20, out_features=20)
        self.linear3 = Linear(in_features=20, out_features=1)
        self.loss_func = nn.MSELoss(reduction='sum')

    def forward(self, X):
        feature = self.linear1(X)
        feature = F.relu(feature)
        feature = self.linear2(feature)
        feature = F.relu(feature)
        out = self.linear3(feature)
        return out

    def nll(self, data, n_samples=1):
        inputs, targets = data['inputs'], data['targets']
        outputs = []
        for _ in range(n_samples):
            out = self(inputs)
            out = torch.squeeze(out, dim=-1)
            outputs.append(out)
        outputs = torch.stack(outputs)
        targets = targets.expand_as(outputs)
        res = self.loss_func(outputs, targets) / n_samples
        return res

class CNN:
    pass


