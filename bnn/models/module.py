import torch
from torch.nn import Module


class BNNModule(Module):
    def __init__(self):
        super().__init__()

    def _kl(self) -> torch.Tensor:
        "Return KL divergence between parameters defined in this module, not children modules"
        return torch.tensor(0, dtype=torch.float32)

    def KL(self) -> torch.Tensor:
        "Calculate the KL divergence between module's prior and current posterior"
        result = self._kl()
        for module in self.children():
            if isinstance(module, BNNModule):
                result = result + module.KL()
        return result
