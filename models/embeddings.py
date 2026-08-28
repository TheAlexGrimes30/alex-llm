import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    Преобразует token ids в dense-вектора.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
    ):
        super().__init__()

        self.weight = nn.Embedding(
            vocab_size,
            d_model,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight(input_ids)

    @property
    def embedding_weight(self):
        return self.weight.weight
