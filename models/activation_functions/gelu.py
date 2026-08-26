import math

import torch
from torch import nn


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit

    Formula:
        GELU(x) = x * Phi(x)

    Equivalent:
        0.5 * x *
        (1 + erf(x / sqrt(2)))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))