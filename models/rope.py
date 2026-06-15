import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Разделяет последний dim на 2 части и меняет их местами с поворотом.
    """

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embedding)
    """

    def __init__(
            self,
            dim: int,
            base: int = 10000,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")

        self.dim = dim

        inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2).float() / dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.Device = None):
        """
        Генерирует cos/sin для всех позиций
        """

        if device is None:
            device = self.inv_freq.device

        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        return cos, sin

    def apply_rope(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> torch.Tensor:

        seq_len = x.size(-2)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        x_rot = rotate_half(x)

        return x * cos + x_rot * sin