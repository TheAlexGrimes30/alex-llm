import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_block import TransformerBlock
from models.rmsnorm import RMSNorm


class LLM(nn.Module):
    """
    Decoder-only Transformer LLM
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_base: int = 10000,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                rope_base=rope_base,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k=top_k,
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight


    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ):
        """
        input_ids: (B, T)
        """

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Autoregressive generation with KV-cache
        """

        self.eval()

        for _ in range(max_new_tokens):

            logits = self.forward(input_ids, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.zeros_like(logits).scatter_(-1, indices, F.softmax(values, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def reset_cache(self):
        for block in self.blocks:
            block.reset_cache()