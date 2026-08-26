import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softplus = F.softplus(x)
        return x * torch.tanh(softplus)

