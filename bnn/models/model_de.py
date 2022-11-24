import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_reg_loss(preds, targets):
    """ Negative log-likelihood loss function. """
    mean, var = preds
    return (torch.log(var) + ((targets - mean).pow(2))/var).sum()    

class SingleModel(nn.Module):
    def __init__(self, cfg):
        super(SingleModel, self).__init__()
        if cfg['name'] == 'MLP':
            self.model = MLP(cfg)
        elif cfg['name'] == 'GaussianMLP':
            self.model = GaussianMLP(cfg)
        else:
            assert(f"Do not support {cfg['model_name']}")

        self.loss_name = cfg['loss']

        if cfg['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif cfg['loss'] == 'bce':
            self.loss = nn.BCELoss()
        elif cfg['loss'] == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif cfg['loss'] == 'softplus':
            self.loss == nn.Softplus()
        elif cfg['loss'] == 'nll_reg':
            self.loss = nll_reg_loss
        else:
            assert(f"Do not support {cfg['loss']}")

    def forward(self, inputs, targets=None):
        preds = self.model(inputs)
        if self.loss_name == 'bce':
            preds = F.sigmoid(preds).squeeze(1)
        if targets is not None:
            loss = self.loss(preds, targets)
            return loss, preds
        else:
            return 0, preds

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

        # if cfg['use_bn']:
        #     self.bn = nn.BatchNorm1d
        # else:
        #     self.bn = nn.Identity()

        module_list = []

        for i in range(0, len(self.net_structure)-1):
            module_list.append(nn.Linear(self.net_structure[i], self.net_structure[i+1]))
            if cfg['use_bn'] and i < len(self.net_structure) - 2:
                module_list.append(nn.BatchNorm1d(self.net_structure[i+1]))
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.module_list[:-1]:
            x = self.act(layer(x))
        x = self.module_list[-1](x)
        return x


class GaussianMLP(MLP):
    """ Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, cfg):
        super(GaussianMLP, self).__init__(cfg)

    def forward(self, x):
        # connect layers
        for layer in self.module_list[:-1]:
            x = self.act(layer(x))
        x = self.module_list[-1](x)

        mean, variance = torch.split(x, self.outputs, dim=1)
        # add softplus and eps for numerical stability
        variance = F.softplus(variance) + 1e-6

        return mean, variance