import torch
from torch import nn


class ReLU(nn.Module):
    """
    ReLU activation

    Formula:
        ReLU(x) = max(0, x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))