import torch
from torch import nn


class LeakyReLU(nn.Module):
    """
    LeakyReLU activation

    Formula:
        x                 if x >= 0
        negative_slope*x  otherwise
    """

    def __init__(self, negative_slope: float = 0.01):

        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x >= 0,
            x,
            self.negative_slope * x,
        )