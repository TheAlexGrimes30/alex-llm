import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    Wrapper over PyTorch SDPA (FlashAttention backend)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        q, k, v: (B, H, T, D)
        attn_mask: optional (B, 1, 1, T) or (B, 1, T, T)
        """

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )