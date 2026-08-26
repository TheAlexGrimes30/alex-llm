import torch
from torch import nn


class GLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.value_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False
        )

        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_proj(x)
        gate = self.gate_proj(x)

        return value * torch.sigmoid(gate)