"""Layer normalization for transformer neural network architectures."""

import torch
from torch import nn


class LayerNormalization(nn.Module):
    """Layer normalization over the last embedding dimension."""

    def __init__(self, embeddings_dimension: int) -> None:
        """Initialize learnable scale and shift parameters for the embedding dimension."""
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embeddings_dimension))
        self.shift = nn.Parameter(torch.zeros(embeddings_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor over the last dimension and apply learnable scale and shift."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
