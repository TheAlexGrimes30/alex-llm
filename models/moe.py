import torch
import torch.nn as nn
import torch.nn.functional as F

from models.activation_functions.swiglu import SwiGLU


class MoE(nn.Module):
    """
    Sparse Mixture of Experts (LLM-style, Top-k routing)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        """

        router_logits = self.router(x)

        router_probs = F.softmax(router_logits, dim=-1)

        # top-k experts
        topk_probs, topk_idx = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1
        )

        # normalize top-k weights
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):

            mask = (topk_idx == expert_id)

            if not mask.any():
                continue

            expert_out = self.experts[expert_id](x)

            weight = (topk_probs * mask.float()).sum(dim=-1, keepdim=True)

            output += expert_out * weight

        return output, router_logits
