from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from models.activation_functions.swiglu import SwiGLU


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_logits: torch.Tensor

class SparseMoE(nn.Module):
    def __init__(
            self,
            d_model: int,
            hidden_dim: int,
            num_experts: int = 8,
            top_k: int = 2,
            shared_expert: bool = True,
    ):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [SwiGLU(d_model, hidden_dim) for _ in range(num_experts)]
        )

        self.shared_expert = SwiGLU(d_model, hidden_dim) if shared_expert else None

    def forward(self, x: torch.Tensor) -> MoEOutput:
        b, t, d = x.shape
        flat_x = x.reshape(-1, d)

        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits.float(), dim=-1)

        topk_probs, topk_idx = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs = topk_probs.to(dtype=x.dtype)

        out = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_idx == expert_id)
            if token_idx.numel() == 0:
                continue

            expert_in = flat_x[token_idx]
            expert_out = expert(expert_in)
            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        if self.shared_expert is not None:
            out = out + self.shared_expert(flat_x)

        hard_assignment = F.one_hot(topk_idx[:, 0], self.num_experts).float()
        tokens_per_expert = hard_assignment.mean(dim=0)
        prob_per_expert = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * prob_per_expert)
        router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()

        return MoEOutput(
            hidden_states=out.view(b, t, d),
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
            router_logits=router_logits.view(b, t, self.num_experts),
        )




