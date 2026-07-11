"""Feed-forward network layer for transformer models."""

import torch
from torch import nn

from tgedr_lm.activations import GELU
from tgedr_lm.configuration import BaseModelConfig


class FeedForward(nn.Module):
    """Transformer feed-forward block with expansion and projection layers."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Initialize feed-forward layers from configuration.

        Parameters
        ----------
        cfg : BaseModelConfig
            Model configuration containing the embedding dimension.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embeddings_dimension, 4 * cfg.embeddings_dimension),
            GELU(),
            nn.Linear(4 * cfg.embeddings_dimension, cfg.embeddings_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after linear-GELU-linear transformation.
        """
        return self.layers(x)
