# from ..module import BNNModule
# from torch import Tensor


# class Linear(BNNModule):
#     def __init__(self, **kwargs):
#         super().__init__()

#     def _kl(self) -> Tensor:
#         pass

#     def _sample_weight(self) -> Tensor:
#         pass

#     def forward(self, input: Tensor) -> Tensor:
#         pass
import torch
import torch.nn as nn
import torch.functional as F

from ..gaussian import Gaussian, ScaleMixtureGaussian


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, pi, sigma_1, sigma_2, device = 'cpu'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)) # do we need to modify range here
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4)) # de we need to modify range here
        self.weight = Gaussian(self.weight_mu, self.weight_rho, device)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, device)
        # Prior distributions

        self.pi = pi
        self.sigma_2 = sigma_1
        self.sigma_2 = sigma_2 

        self.device = device

        self.weight_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.bias_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)