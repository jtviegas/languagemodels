import torch
import torch.nn as nn

from tgedr_languagemodels.layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            embedding_dimension=cfg["emb_dim"],
            output_dimension=cfg["emb_dim"],
            context_length=cfg["context_length"],
            heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
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


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for transformer networks."""

    def __init__(
        self,
        embedding_dimension: int,
        output_dimension: int,
        context_length: int,
        heads: int,
        qkv_bias: bool = False,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the MultiHeadAttention module.

        Parameters
        ----------
        embedding_dimension : int
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
        assert output_dimension % heads == 0, "output_dimension must be divisible by heads"

        self.embedding_dimension = embedding_dimension
        self.output_dimension = output_dimension
        self.context_length = context_length
        self.head_dimension = output_dimension // heads
        self.heads = heads

        self.W_query = nn.Linear(embedding_dimension, output_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(embedding_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(embedding_dimension, output_dimension, bias=qkv_bias)

        self.out_projection = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        n_sequences, sequence_length, embedding_dimension = x.shape
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
