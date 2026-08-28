import torch
from torch import nn

from models.attention.attention_utils import KVCache, build_attention_mask, merge_heads
from models.attention.flash_attention import SDPAttention


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            rope: nn.Module | None = None,
            dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.rope = rope
        self.dropout = dropout

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.attention = SDPAttention()

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, D)
            ->
        (B, H, T, Hd)
        """

        B, T, _ = x.shape

        return (
            x.view(
                B,
                T,
                self.n_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            past_key_value: KVCache | None = None,
            use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:

        B, T, _ = x.shape

        past_len = (
            past_key_value[0].size(2)
            if past_key_value is not None
            else 0
        )

        q = self._reshape(
            self.q_proj(x)
        )

        k = self._reshape(
            self.k_proj(x)
        )

        v = self._reshape(
            self.v_proj(x)
        )

        if self.rope is not None:
            cos, sin = self.rope(
                seq_len=T,
                device=x.device,
                offset=past_len,
            )

            q = self.rope.apply_rope(
                q,
                cos,
                sin,
            )

            k = self.rope.apply_rope(
                k,
                cos,
                sin,
            )

        if past_key_value is not None:
            past_k, past_v = past_key_value

            k = torch.cat(
                [past_k, k],
                dim=2,
            )

            v = torch.cat(
                [past_v, v],
                dim=2,
            )

        present_key_value = (
            (k, v)
            if use_cache
            else None
        )

        effective_mask, is_causal = (
            build_attention_mask(
                x=x,
                T=T,
                past_len=past_len,
                attn_mask=attn_mask,
            )
        )

        out = self.attention(
            q,
            k,
            v,
            attn_mask=effective_mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=is_causal,
        )

        out = merge_heads(out)

        return (
            self.o_proj(out),
            present_key_value,
        )
