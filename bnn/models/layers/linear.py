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
                 device=None, dtype=torch.float32,
                 logstd=(0,), mixture_weights=(1,)) -> None:
        assert len(logstd) == len(mixture_weights)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mixture_len = len(mixture_weights)
        std = torch.exp(torch.tensor(logstd, **factory_kwargs))
        weight_prior_mean = torch.zeros((out_features, in_features, mixture_len), **factory_kwargs) * std
        weight_prior_std = torch.ones((out_features, in_features, mixture_len), **factory_kwargs) * std
        mixture_weights = torch.tensor(mixture_weights, **factory_kwargs)
        self.register_buffer('weight_prior_mean', weight_prior_mean, False)
        self.register_buffer('weight_prior_std', weight_prior_std, False)
        self.register_buffer('mixture_weights', mixture_weights, False)

        self.weight_posteiror_mean = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_posteiror_rho = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            bias_prior_mean = torch.zeros((out_features, mixture_len), **factory_kwargs) * std
            bias_prior_std = torch.ones((out_features, mixture_len), **factory_kwargs) * std
            self.register_buffer('bias_prior_mean', bias_prior_mean, False)
            self.register_buffer('bias_prior_std', bias_prior_std, False)

            self.bias_posteiror_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_posteiror_rho = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.use_bias = bias
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_posteiror_mean, a=math.sqrt(5))
        # nn.init.normal_(self.weight_posteiror_mean, 0, 0.1)
        nn.init.constant_(self.weight_posteiror_rho, -3)
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_posteiror_mean)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_posteiror_mean, -bound, bound)
            nn.init.constant_(self.bias_posteiror_rho, -3)

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

    def _get_prior(self) -> Dict[str, D.Distribution]:
        result = dict()

        if len(self.mixture_weights) == 1:
            weight_prior = D.Normal(self.weight_prior_mean.squeeze(-1), self.weight_prior_std.squeeze(-1))
        else:
            weight_prior_mix = D.Categorical(self.mixture_weights.expand(self.out_features, self.in_features, -1))
            weight_prior_comp = D.Normal(self.weight_prior_mean, self.weight_prior_std)
            weight_prior = D.MixtureSameFamily(weight_prior_mix, weight_prior_comp)
        result['weight'] = D.Independent(weight_prior, 2)

        if self.use_bias:
            if len(self.mixture_weights) == 1:
                bias_prior = D.Normal(self.bias_prior_mean.squeeze(-1), self.bias_prior_std.squeeze(-1))
            else:
                bias_prior_mix = D.Categorical(self.mixture_weights.expand(self.out_features, -1))
                bias_prior_comp = D.Normal(self.bias_prior_mean, self.bias_prior_std)
                bias_prior = D.MixtureSameFamily(bias_prior_mix, bias_prior_comp)
            result['bias'] = D.Independent(bias_prior, 1)
        
        return result

    def _kl(self, n_samples: int) -> Tensor:
        posterior = self._get_posterior()
        prior = self._get_prior()

        if len(self.mixture_weights) == 1:
            kl = D.kl_divergence(posterior['weight'], prior['weight'])
            if self.use_bias:
                kl = kl + D.kl_divergence(posterior['bias'], prior['bias'])
        
        else:
            kl_sample = []
            for _ in range(n_samples):
                weight = posterior['weight'].rsample()
                kl_sample.append(posterior['weight'].log_prob(weight) - prior['weight'].log_prob(weight))
            kl = torch.stack(kl_sample).mean()
            if self.use_bias:
                kl_sample = []
                for _ in range(n_samples):
                    bias = posterior['bias'].rsample()
                    kl_sample.append(posterior['bias'].log_prob(bias) - prior['bias'].log_prob(bias))
                kl_bias = torch.stack(kl_sample).mean()
                kl += kl_bias

        return kl

    def forward(self, input: Tensor) -> Tensor:
        posterior = self._get_posterior()
        weight = posterior['weight'].rsample()
        # weight = self.weight_posteiror_mean
        if self.use_bias:
            bias = posterior['bias'].rsample()
            # bias = self.bias_posteiror_mean
        else:
            bias = None
        return F.linear(input, weight, bias)