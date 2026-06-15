import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            rope: nn.Module = None,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.dropout = dropout
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

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
            cos: torch.Tensor = None,
            sin: torch.Tensor = None,
            use_cache: bool = False,
    ):
        """
        x: (B, T, D)
        """

        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        if self.rope is not None:
            cos, sin = self.rope(T, device=x.device)

            q = self.rope.apply_rope(q, cos, sin)
            k = self.rope.apply_rope(k, cos, sin)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)

            k = self.k_cache
            v = self.v_cache

        out = flash_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = self._merge(out)
        return self.o_proj(out)
