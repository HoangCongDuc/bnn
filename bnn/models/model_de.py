import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleModel(nn.Module):
    def __init__(self, cfg):
        super(SingleModel, self).__init__()
        if cfg['name'] == 'MLP':
            self.model = MLP(cfg)
        elif cfg['name'] == 'GaussianMLP':
            pass
        else:
            assert(f"Do not support {cfg['model_name']}")

        if cfg['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif cfg['loss'] == 'softmax':
            self.loss = nn.CrossEntropyLoss()
        elif cfg['loss'] == 'softplus':
            self.loss == nn.Softplus()
        else:
            assert(f"Do not support {cfg['loss']}")

    def forward(self, inputs, targets):
        preds = self.model(inputs)
        loss = self.loss(preds, targets)
        return loss, preds

class MLP(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch for regression tasks.

    Attributes:
        inputs (int): inputs of the network
        outputs (int): outputs of the network
        hidden_layers (list): layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
        activation (string): activation function used ('relu', 'tanh' or 'sigmoid')

    """

    def __init__(self, cfg):
        super().__init__()
        super(MLP, self).__init__()
        self.net_structure = cfg['layers']
        
        if cfg['act'] == 'relu':
            self.act = torch.relu
        elif cfg['act'] == 'tanh':
            self.act = torch.tanh
        elif cfg['act'] == 'sigmoid':
            self.act = torch.sigmoid
        else:
            assert('Use "relu","tanh" or "sigmoid" as activation.')

        module_list = []

        for i in range(0, len(self.net_structure)-1):
            module_list.append(nn.Linear(self.net_structure[i], self.net_structure[i+1]))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        for layer in self.module_list[:-1]:
            x = self.act(layer(x))
        x = self.module_list[-1](x)
        return x


# class GaussianMLP(MLP):
#     """ Gaussian MLP which outputs are mean and variance.

#     Attributes:
#         inputs (int): number of inputs
#         outputs (int): number of outputs
#         hidden_layers (list of ints): hidden layer sizes

#     """

#     def __init__(self, cfg):
#         super(GaussianMLP, self).__init__(cfg)
#         self.in_channels = cfg.in_channels
#         self.out_channels = cfg.out_channels
#     def forward(self, x):
#         # connect layers
#         for i in range(self.nLayers):
#             layer = getattr(self, 'layer_'+str(i))
#             x = self.act(layer(x))
#         layer = getattr(self, 'layer_' + str(self.nLayers))
#         x = layer(x)
#         mean, variance = torch.split(x, self.outputs, dim=1)
#         variance = F.softplus(variance) + 1e-6
#         return mean, variance

# class GaussianMixtureMLP(nn.Module):
#     """ Gaussian mixture MLP which outputs are mean and variance.

#     Attributes:
#         models (int): number of models
#         inputs (int): number of inputs
#         outputs (int): number of outputs
#         hidden_layers (list of ints): hidden layer sizes

#     """
#     def __init__(self, cfg):
#         super(GaussianMixtureMLP, self).__init__()
#         self.num_models = cfg.num_models
#         self.in_channels = cfg.in_channels
#         self.out_channels = cfg.out_channels
#         self.hidden_layers = cfg.layers
#         self.activation = cfg.activation
#         for i in range(self.num_models):
#             model = GaussianMLP(cfg)
#             setattr(self, 'model_'+str(i), model)
            
#     def forward(self, x):
#         # connect layers
#         means = []
#         variances = []
#         for i in range(self.num_models):
#             model = getattr(self, 'model_' + str(i))
#             mean, var = model(x)
#             means.append(mean)
#             variances.append(var)
#         means = torch.stack(means)
#         mean = means.mean(dim=0)
#         variances = torch.stack(variances)
#         variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
#         return mean, variance