"""Transformer block implementation for language models.

This module provides the TransformerBlock class, which combines multi-head attention
and feed-forward layers with residual connections and layer normalization.
"""

from torch import nn
import torch

from tgedr_languagemodels.configuration import BaseModelConfig
from tgedr_languagemodels.layers.attention import MultiHeadAttention
from tgedr_languagemodels.layers.feed_forward import FeedForward
from tgedr_languagemodels.layers.normalization import LayerNormalization


class TransformerBlock(nn.Module):
    """A transformer block combining multi-head attention and feed-forward layers.

    This block implements the standard transformer architecture with pre-normalization,
    residual connections, and dropout for regularization.

    Attributes
    ----------
    att : MultiHeadAttention
        Multi-head attention layer.
    ff : FeedForward
        Feed-forward network layer.
    norm1 : LayerNormalization
        Layer normalization before attention.
    norm2 : LayerNormalization
        Layer normalization before feed-forward.
    drop_shortcut : nn.Dropout
        Dropout layer for residual connections.

    """

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Initialize the transformer block with configuration.

        Parameters
        ----------
        cfg : BaseModelConfig
            Configuration object containing model hyperparameters.

        """
        super().__init__()
        self.att = MultiHeadAttention(
            embeddings_dimension=cfg.embeddings_dimension,
            output_dimension=cfg.embeddings_dimension,
            context_length=cfg.context_length,
            heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNormalization(cfg.embeddings_dimension)
        self.norm2 = LayerNormalization(cfg.embeddings_dimension)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after attention and feed-forward layers with residual connections.

        """
        # 1
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 2

        shortcut = x  # 3
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 4
        return x
