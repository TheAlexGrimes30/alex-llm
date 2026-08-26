import torch
from torch import nn


class Tanh(nn.Module):
    """
    Tanh activation

    Formula:
        tanh(x) =
            (exp(x) - exp(-x))
            /
            (exp(x) + exp(-x))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_x = torch.exp(x)
        exp_neg_x = torch.exp(-x)

        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
