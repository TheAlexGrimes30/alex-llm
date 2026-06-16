import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.moe import MoE
from models.rmsnorm import RMSNorm
from models.rope import RotaryEmbedding
from models.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Transformer Block
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            hidden_dim: int,
            dropout: float = 0.0,
            rope_base: int = 10000,
            use_moe: bool = False,
            num_experts: int = 8,
            top_k: int = 2,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)

        self.rope = RotaryEmbedding(
            dim=d_model // n_heads,
            base=rope_base,
        )

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            rope=self.rope,
            dropout=dropout,
        )

        self.use_moe = use_moe

        if use_moe:
            self.feed_forward = MoE(
                d_model=d_model,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
            )
        else:
            self.feed_forward = SwiGLU(
                d_model=d_model,
                hidden_dim=hidden_dim,
            )

    def reset_cache(self):
        """
        Reset KV cache for generation.
        """

        self.attention.reset_cache()

    def forward(
            self,
            x: torch.Tensor,
            use_cache: bool = False,
    ):
        router_logits = None
        residual = x

        x = self.attn_norm(x)

        x = self.attention(
            x,
            use_cache=use_cache,
        )

        x = residual + x

        residual = x

        x = self.ffn_norm(x)

        if self.use_moe:

            x, router_logits = self.feed_forward(x)

        else:

            x = self.feed_forward(x)

        x = residual + x

        return x, router_logits
    