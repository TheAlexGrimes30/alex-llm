from typing import TypeAlias

import torch

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, T, Hd)
        ->
    (B, T, H * Hd)
    """

    B, H, T, Hd = x.shape

    return (
        x.transpose(1, 2)
        .contiguous()
        .view(B, T, H * Hd)
    )

def build_attention_mask(
    x: torch.Tensor,
    T: int,
    past_len: int,
    attn_mask: torch.Tensor | None,
):
    """
    Создаёт causal mask при использовании KV-cache.
    """

    # Обычный pretrain / полный prompt.
    if past_len == 0:
        return attn_mask, True

    # Генерация одного нового токена.
    # Все K находятся в прошлом или в текущей позиции.
    if T == 1:
        return attn_mask, False

    # Несколько новых токенов + KV-cache.
    q_positions = torch.arange(
        past_len,
        past_len + T,
        device=x.device,
    )

    k_positions = torch.arange(
        past_len + T,
        device=x.device,
    )

    causal_mask = (
        k_positions.unsqueeze(0)
        <= q_positions.unsqueeze(1)
    )

    causal_mask = causal_mask.view(
        1,
        1,
        T,
        past_len + T,
    )

    if attn_mask is None:
        return causal_mask, False

    if attn_mask.dtype == torch.bool:
        return attn_mask & causal_mask, False

    additive_mask = torch.zeros(
        1,
        1,
        T,
        past_len + T,
        device=x.device,
        dtype=x.dtype,
    )

    additive_mask = additive_mask.masked_fill(
        ~causal_mask,
        float("-inf"),
    )

    return attn_mask + additive_mask, False
