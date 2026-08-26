import torch
from torch import nn


class SiLU(nn.Module):
    """
    SiLU / Swish activation

    Formula:
        SiLU(x) = x * sigmoid(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid = (
                1.0
                / (
                        1.0
                        + torch.exp(-x)
                )
        )

        return x * sigmoid

