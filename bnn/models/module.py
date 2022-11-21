import torch
import torch.nn as nn


__all__ = ['BNNModule', 'ModuleList']

class BNNModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _kl(self, n_samples: int) -> torch.Tensor:
        "Return KL divergence between parameters defined in this module, not children modules"
        return 0

    def KL(self, n_samples: int = 1) -> torch.Tensor:
        "Calculate the KL divergence between module's prior and current posterior"
        result = self._kl(n_samples)
        for module in self.children():
            if isinstance(module, BNNModule):
                result = result + module.KL(n_samples)
        return result


class ModuleList(nn.ModuleList, BNNModule): pass