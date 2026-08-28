from typing import TypeAlias

import torch
import torch.nn as nn

from models.attention.flash_attention import SDPAttention

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]

class MultiHeadAttention(nn.Module):
    """
    Decoder-only self-attention с поддержкой:

    - MHA
    - GQA
    - MQA
    - RoPE
    - KV-cache
    - PyTorch SDPA
    - causal self-attention

    Режимы по n_kv_heads:

        n_kv_heads == n_heads
            -> MHA

        1 < n_kv_heads < n_heads
            -> GQA

        n_kv_heads == 1
            -> MQA
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int | None = None,
            rope: nn.Module | None = None,
            dropout: float = 0.0,
            attention_backend: nn.Module | None = None,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        if n_kv_heads is None:
            n_kv_heads = n_heads

        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")

        if n_heads % n_kv_heads != 0:
            raise ValueError(
                "n_heads must be divisible by n_kv_heads"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE"
            )

        self.dropout = dropout
        self.rope = rope

        self.q_proj = nn.Linear(
            d_model,
            n_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            n_heads * self.head_dim,
            d_model,
            bias=False,
        )

        self.attention_backend = (
            attention_backend
            if attention_backend is not None
            else SDPAttention()
        )

    def _reshape_q(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        (B, T, n_heads * head_dim)
            ->
        (B, n_heads, T, head_dim)
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

    def _reshape_kv(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        (B, T, n_kv_heads * head_dim)
            ->
        (B, n_kv_heads, T, head_dim)
        """

        B, T, _ = x.shape

        return (
            x.view(
                B,
                T,
                self.n_kv_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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

    def _repeat_kv(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Повторяет KV heads для GQA/MQA.

        Input:
            x: (B, n_kv_heads, T, Hd)

        Output:
            (B, n_heads, T, Hd)
        """

        if self.n_rep == 1:
            return x

        B, Hkv, T, Hd = x.shape

        if Hkv * self.n_rep != self.n_heads:
            raise ValueError(
                f"Invalid head configuration: "
                f"{Hkv} * {self.n_rep} != {self.n_heads}"
            )

        # (B, Hkv, T, Hd)
        # ->
        # (B, Hkv, 1, T, Hd)
        x = x.unsqueeze(2)

        # (B, Hkv, 1, T, Hd)
        # ->
        # (B, Hkv, n_rep, T, Hd)
        x = x.expand(
            B,
            Hkv,
            self.n_rep,
            T,
            Hd,
        )

        # (B, Hkv, n_rep, T, Hd)
        # ->
        # (B, n_heads, T, Hd)
        x = x.reshape(
            B,
            self.n_heads,
            T,
            Hd,
        )

        return x

    @staticmethod
    def _validate_cache(
            past_key_value: KVCache,
            batch_size: int,
            n_kv_heads: int,
            head_dim: int,
    ) -> None:
        past_k, past_v = past_key_value

        if past_k.shape != past_v.shape:
            raise ValueError(
                "past_k and past_v must have the same shape"
            )

        if past_k.ndim != 4:
            raise ValueError(
                "past_key_value tensors must have shape "
                "(B, n_kv_heads, T, head_dim)"
            )

        if past_k.size(0) != batch_size:
            raise ValueError(
                "KV cache batch size does not match input batch size"
            )

        if past_k.size(1) != n_kv_heads:
            raise ValueError(
                "KV cache n_kv_heads does not match attention config"
            )

        if past_k.size(-1) != head_dim:
            raise ValueError(
                "KV cache head_dim does not match attention config"
            )


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        KVCache | None,
    ]:
        """
        Args:
            x:
                (B, T, D)

            attn_mask:
                Опциональная маска.

            past_key_value:
                tuple(
                    past_k,
                    past_v,
                )

                каждый:
                    (B, n_kv_heads, T_past, head_dim)

            use_cache:
                Вернуть ли текущий KV-cache.

        Returns:
            output:
                (B, T, D)

            present_key_value:
                None либо tuple(k, v)

        Важное ограничение:
            если past_key_value передан, наиболее простой и надёжный
            inference-сценарий — подавать только новые токены.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape (B, T, D), got {tuple(x.shape)}"
            )

        B, T, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {D}"
            )

        if T <= 0:
            raise ValueError(
                "Sequence length must be > 0"
            )

        past_len = 0

        if past_key_value is not None:
            self._validate_cache(
                past_key_value,
                batch_size=B,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )

            past_len = past_key_value[0].size(2)

        q = self._reshape_q(
            self.q_proj(x)
        )

        k = self._reshape_kv(
            self.k_proj(x)
        )

        v = self._reshape_kv(
            self.v_proj(x)
        )

        # RoPE должен знать абсолютную позицию новых токенов.
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
                [past_k.to(k.device), k],
                dim=2,
            )

            v = torch.cat(
                [past_v.to(v.device), v],
                dim=2,
            )

        present_key_value = (
            (k, v)
            if use_cache
            else None
        )

        k_for_attn = self._repeat_kv(k)
        v_for_attn = self._repeat_kv(v)

        if past_len == 0:
            is_causal = True
            effective_mask = attn_mask

        elif T == 1:
            is_causal = False
            effective_mask = attn_mask

        else:
            is_causal = False

            q_positions = torch.arange(
                past_len,
                past_len + T,
                device=x.device,
            )

            k_positions = torch.arange(
                0,
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
                effective_mask = causal_mask
            else:
                if attn_mask.dtype == torch.bool:
                    effective_mask = (
                        attn_mask
                        & causal_mask
                    )
                else:
                    additive_causal_mask = torch.zeros(
                        (
                            1,
                            1,
                            T,
                            past_len + T,
                        ),
                        device=x.device,
                        dtype=x.dtype,
                    )

                    additive_causal_mask = (
                        additive_causal_mask.masked_fill(
                            ~causal_mask,
                            float("-inf"),
                        )
                    )

                    effective_mask = (
                        attn_mask
                        + additive_causal_mask
                    )

        out = self.attention_backend(
            q,
            k_for_attn,
            v_for_attn,
            attn_mask=effective_mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=is_causal,
        )

        out = self._merge_heads(out)

        return (
            self.o_proj(out),
            present_key_value,
        )
