import torch
import torch.nn as nn

from models.flash_attention import FlashAttention


class MultiHeadAttention(nn.Module):
    """
    LLM-grade Multi-Head Attention with:
    - RoPE
    - KV cache
    - FlashAttention backend
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: nn.Module = None,
        dropout: float = 0.0,
        flash_attn: nn.Module = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.dropout = dropout
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.flash_attn = flash_attn if flash_attn is not None else FlashAttention()

        self.k_cache = None
        self.v_cache = None

    def _reshape(self, x: torch.Tensor):
        """
        (B, T, D) -> (B, H, T, Hd)
        """
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor):
        """
        (B, H, T, Hd) -> (B, T, D)
        """
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)


    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, D)
        """

        B, T, D = x.shape
        device = x.device

        if self.training:
            self.reset_cache()

        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))

        if self.rope is not None:
            cos, sin = self.rope(T, device=device)
            q = self.rope.apply_rope(q, cos, sin)
            k = self.rope.apply_rope(k, cos, sin)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)

            k = self.k_cache.to(device)
            v = self.v_cache.to(device)

        out = self.flash_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = self._merge_heads(out)
        return self.o_proj(out)