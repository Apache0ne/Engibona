from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GroupQuantizedEmbedding


class TiedGroupQuantizedLMHead(nn.Module):
    """LM head that reuses one quantized embedding state exactly.

    Qwen3-VL dense checkpoints may tie token embeddings and the LM head. A
    converter must not silently create two independently recovered codebooks
    when the source architecture contains one shared parameter.
    """

    def __init__(self, embedding: GroupQuantizedEmbedding) -> None:
        super().__init__()
        # Avoid registering the same quantization state twice as a child module.
        object.__setattr__(self, "_embedding", embedding)
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        self.bias = None

    @property
    def embedding(self) -> GroupQuantizedEmbedding:
        return object.__getattribute__(self, "_embedding")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding._surrogate().to(x.dtype))

    @torch.no_grad()
    def hard_surrogate(self) -> torch.Tensor:
        return self.embedding.hard_surrogate()

    def hard_codes_and_scales(self):
        return self.embedding.hard_codes_and_scales()

    def regularization_loss(self) -> torch.Tensor:
        # The shared state is regularized through the embedding exactly once.
        return torch.zeros((), device=self.embedding.latent_weight.device)
