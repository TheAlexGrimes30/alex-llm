import torch
from torch import nn


class Softplus(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
                torch.maximum(
                    x,
                    torch.zeros_like(x),
                )
                +
                torch.log1p(
                    torch.exp(-torch.abs(x))
                )
        )