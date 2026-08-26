import torch
from torch import nn


class Sigmoid(nn.Module):
    """
    Sigmoid activation

    Formula:
        sigmoid(x) = 1 / (1 + exp(-x))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  1.0 / (1.0 + torch.exp(-x))