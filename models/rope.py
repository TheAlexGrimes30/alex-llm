import torch
import torch.nn as nn
from torch import Tensor


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Разделяет последний dim на 2 части и меняет их местами с поворотом.
    """

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

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

        inv_freq = 1.0 / (
                base
                ** (
                        torch.arange(
                            0,
                            dim,
                            2,
                            dtype=torch.float32,
                        )
                        / dim
                )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
            self,
            seq_len: int,
            device: torch.device | None = None,
            offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Генерирует cos/sin для позиций:
            offset ... offset + seq_len - 1

        Returns:
            cos: (seq_len, dim)
            sin: (seq_len, dim)
        """

        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        if offset < 0:
            raise ValueError("offset must be >= 0")

        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(
            offset,
            offset + seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(
            positions,
            self.inv_freq.to(device=device),
        )

        emb = torch.cat(
            [freqs, freqs],
            dim=-1,
        )

        return emb.cos(), emb.sin()

    def apply_rope(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Применяет RoPE к tensor формы:
            (B, H, T, D)

        cos/sin:
            (T, D)
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected x shape (B, H, T, D), got {tuple(x.shape)}"
            )

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got {x.size(-1)}"
            )

        seq_len = x.size(-2)

        if cos.size(0) < seq_len or sin.size(0) < seq_len:
            raise ValueError(
                "cos/sin sequence length is smaller than x sequence length"
            )

        cos = cos[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = sin[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return (
                x * cos
                + rotate_half(x) * sin
        )