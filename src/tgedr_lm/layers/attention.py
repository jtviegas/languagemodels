"""Attention mechanisms for transformer networks.

This module provides multi-head attention components used in transformer
architectures, enabling models to focus on different parts of the input
sequence simultaneously.
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for transformer networks."""

    def __init__(
        self,
        embeddings_dimension: int,
        output_dimension: int,
        context_length: int,
        heads: int,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        dropout: float = 0.5,
    ) -> None:
        """Initialize the MultiHeadAttention module.

        Parameters
        ----------
        embeddings_dimension : int
            Dimension of the input embeddings.
        output_dimension : int
            Dimension of the output.
        context_length : int
            Maximum sequence length.
        heads : int
            Number of attention heads.
        qkv_bias : bool, optional
            Whether to use bias in query, key, value projections (default: False).
        dropout : float, optional
            Dropout rate (default: 0.5).
        """
        super().__init__()
        if output_dimension % heads != 0:
            raise ValueError("output_dimension must be divisible by heads")  # noqa: EM101, TRY003

        self.embedding_dimension = embeddings_dimension
        self.output_dimension = output_dimension
        self.context_length = context_length
        self.head_dimension = output_dimension // heads
        self.heads = heads

        self.W_query = nn.Linear(embeddings_dimension, output_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(embeddings_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(embeddings_dimension, output_dimension, bias=qkv_bias)

        self.out_projection = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_sequences, sequence_length, embeddings_dimension).

        Returns
        -------
        torch.Tensor
            Output context vectors of shape (n_sequences, sequence_length, output_dimension).
        """
        n_sequences, sequence_length, embeddings_dimension = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(n_sequences, sequence_length, self.heads, self.head_dimension)
        values = values.view(n_sequences, sequence_length, self.heads, self.head_dimension)
        queries = queries.view(n_sequences, sequence_length, self.heads, self.head_dimension)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # Transposes from shape (n_sequences, sequence_length, heads, head_dimension) to (n_sequences, heads, sequence_length, head_dimension)

        attention_scores = queries @ keys.transpose(-2, -1)
        masked_attention_scores = attention_scores.masked_fill(
            self.mask.bool()[:sequence_length, :sequence_length], -torch.inf
        )
        attn_weights = torch.softmax(masked_attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights_after_dropout = self.dropout(attn_weights)

        context_vec = self.out_projection(
            (attn_weights_after_dropout @ values)
            .transpose(1, 2)
            .contiguous()
            .view(n_sequences, sequence_length, self.output_dimension)
        )
        return context_vec
