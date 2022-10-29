import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from ..module import BNNModule


__all__ = ['Linear']

class Linear(BNNModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_prior_mean = torch.zeros((out_features, in_features), **factory_kwargs)
        self.weight_prior_std = torch.ones((out_features, in_features), **factory_kwargs)
        self.register_buffer('weight_prior_mean', self.weight_prior_mean, False)
        self.register_buffer('weight_prior_std', self.weight_prior_std, False)

        self.weight_posteiror_mean = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_posteiror_rho = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias_prior_mean = torch.zeros(out_features, **factory_kwargs)
            self.bias_prior_std = torch.ones(out_features, **factory_kwargs)
            self.register_buffer('bias_prior_mean', self.bias_prior_mean, False)
            self.register_buffer('bias_prior_std', self.bias_prior_std, False)

            self.bias_posteiror_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_posteiror_rho = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.use_bias = bias
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_posteiror_mean, a=math.sqrt(5))
        nn.init.normal_(self.weight_posteiror_rho, mean=-3, std=0.1)
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_posteiror_mean)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_posteiror_mean, -bound, bound)
            nn.init.normal_(self.bias_posteiror_rho, mean=-3, std=0.1)

    def _get_posterior(self) -> Dict[str, D.Distribution]:
        result = dict()

        weight_posterior_std = F.softplus(self.weight_posteiror_rho)
        weight = D.Independent(D.Normal(self.weight_posteiror_mean, weight_posterior_std), 2)
        result['weight'] = weight
        
        if self.use_bias:
            bias_posterior_std = F.softplus(self.bias_posteiror_rho)
            bias = D.Independent(D.Normal(self.bias_posteiror_mean, bias_posterior_std), 1)
            result['bias'] = bias
        
        return result

    def _kl(self) -> Tensor:
        posterior = self._get_posterior()
        weight_prior = D.Independent(D.Normal(self.weight_prior_mean, self.weight_prior_std), 2)
        kl = D.kl_divergence(posterior['weight'], weight_prior)
        if self.use_bias:
            bias_prior = D.Independent(D.Normal(self.bias_prior_mean, self.bias_prior_std), 1)
            kl = kl + D.kl_divergence(posterior['bias'], bias_prior)
        return kl

    def forward(self, input: Tensor) -> Tensor:
        posterior = self._get_posterior()
        weight = posterior['weight'].rsample()
        if self.use_bias:
            bias = posterior['bias'].rsample()
        else:
            bias = None
        return F.linear(input, weight, bias)