import torch
from torch import nn
import torch.nn.functional as F

class ReGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ):

        super().__init__()

        self.w1 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.w2 = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

        self.w3 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)

        hidden = (
                F.relu(gate)
                * up
        )

        return self.x2(hidden)