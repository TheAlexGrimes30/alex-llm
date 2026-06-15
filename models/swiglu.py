import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU activation

    Formula:
        SwiGLU(x) = Swish(xW1) ⊙ (xW3)   where Swish(x) = x * sigmoid(x)
    """

    def __init__(
            self,
            d_model: int,
            hidden_dim: int = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """

        gate = self.w1(x)
        up = self.w3(x)
        hidden = F.silu(gate) * up

        return self.w2(hidden)
    