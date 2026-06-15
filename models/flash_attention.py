import torch
import torch.nn.functional as F


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = True,
):
    """
    FlashAttention backend wrapper over PyTorch SDPA

    q, k, v: (B, H, T, D)
    """

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )