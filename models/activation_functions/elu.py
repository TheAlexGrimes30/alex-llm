import torch
from torch import nn


class ELU(nn.Module):
    """
    ELU activation

    Formula:
        x                       if x > 0
        alpha * (exp(x) - 1)   otherwise
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1.0))