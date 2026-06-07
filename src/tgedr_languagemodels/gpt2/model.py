"""GPT model definition and supporting layers for language modeling."""

import torch
from torch import nn

from tgedr_languagemodels.configuration import BaseModelConfig
from tgedr_languagemodels.layers.blocks import TransformerBlock
from tgedr_languagemodels.layers.normalization import LayerNormalization


class GPT2Model(nn.Module):
    """GPT-2 model for language modeling.

    Attributes
    ----------
    tok_emb : nn.Embedding
        Token embedding layer.
    pos_emb : nn.Embedding
        Positional embedding layer.
    drop_emb : nn.Dropout
        Dropout layer for embeddings.
    trf_blocks : nn.Sequential
        Stack of transformer blocks.
    final_norm : LayerNormalization
        Final layer normalization.
    out_head : nn.Linear
        Output linear layer for logits.

    Methods
    -------
    forward(in_idx: torch.Tensor) -> torch.Tensor
        Forward pass that returns logits for input token indices.
    """

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Initialize GPT-2 model layers from the given configuration.

        Parameters
        ----------
        cfg : BaseModelConfig
            Model configuration containing vocabulary size, context length,
            embedding dimension, dropout rate, and number of transformer layers.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocabulary_size, cfg.embeddings_dimension)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embeddings_dimension)

        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNormalization(cfg.embeddings_dimension)
        self.out_head = nn.Linear(cfg.embeddings_dimension, cfg.vocabulary_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for the given input token indices.

        Parameters
        ----------
        in_idx : torch.Tensor
            Integer tensor of shape (batch_size, seq_len) with token indices.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, seq_len, vocabulary_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # relational positional embedding layer for sequence length up to context_length,
        # with embedding dimension emb_dim
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
