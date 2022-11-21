import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from .layers import Linear

class MLP(BNNModule):
    def __init__(self, cfg):
        super().__init__()
        logstd = cfg['log_std']
        mixture_weights = cfg['mixture_weight']

        self.layers = ModuleList()
        for i in range(len(cfg['layers']) - 1):
            layer = Linear(in_features=cfg['layers'][i],
                           out_features=cfg['layers'][i + 1],
                           logstd=logstd, mixture_weights=mixture_weights)
            self.layers.append(layer)

        if cfg['nll_loss'] == 'mse':
            self.loss_func = nn.MSELoss(reduction='sum')
        elif cfg['nll_loss'] == 'cross_entropy':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, X, sample=True):
        feats = torch.flatten(X, 1)
        for layer in self.layers[:-1]:
            feats = layer(feats, sample)
            feats = F.relu(feats)
        out = self.layers[-1](feats, sample)
        return out

    def nll(self, data, n_samples: int = 1):
        assert n_samples >= 0
        inputs, targets = data['inputs'], data['targets']
        if n_samples == 0:
            out = self(inputs, sample=False)
            return self.loss_func(out, targets)
        res = 0
        for _ in range(n_samples):
            out = self(inputs)
            res = res + self.loss_func(out, targets)
        res = res / n_samples
        return res

class CNN:
    pass


