from ..module import BNNModule
from torch import Tensor


class Linear(BNNModule):
    def __init__(self, **kwargs):
        super().__init__()

    def _kl(self) -> Tensor:
        pass

    def _sample_weight(self) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        pass