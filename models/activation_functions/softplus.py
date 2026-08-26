import torch
from torch import nn


class Softplus(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x))