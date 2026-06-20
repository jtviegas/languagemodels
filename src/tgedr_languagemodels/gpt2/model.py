"""GPT model definition and supporting layers for language modeling."""

import torch
from torch import nn
import numpy as np

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

    @staticmethod
    def assign(left, right) -> torch.nn.Parameter:
        """Validate shapes and return right as a new Parameter with the same shape as left.

        Parameters
        ----------
        left : torch.Tensor
            Reference tensor whose shape must match right.
        right : array-like
            Source data to wrap as a parameter.

        Returns
        -------
        torch.nn.Parameter
            Parameter wrapping the values of right.

        Raises
        ------
        ValueError
            If left and right have different shapes.
        """
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {{right.shape}}")  # noqa: EM102, TRY003
        return torch.nn.Parameter(torch.tensor(right))

    def load_weights(self, params: dict) -> None:
        """Load pretrained weights into the model from a parameters dictionary.

        Parameters
        ----------
        params : dict
            Dictionary containing pretrained weights for token embeddings, positional
            embeddings, transformer blocks, and output layer.
        """
        # 1 Sets the model's positional and token embedding weights to those specified in params.

        self.pos_emb.weight = self.assign(self.pos_emb.weight, params["wpe"])
        self.tok_emb.weight = self.assign(self.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):  # 2 Iterates over each transformer block in the model
            # 3 The np.split function is used to divide the attention and bias weights
            # into three equal parts for the query, key, and value components.
            q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.weight = self.assign(self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = self.assign(self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = self.assign(self.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = self.assign(self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = self.assign(self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = self.assign(self.trf_blocks[b].att.W_value.bias, v_b)

            self.trf_blocks[b].att.out_projection.weight = self.assign(
                self.trf_blocks[b].att.out_projection.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].att.out_projection.bias = self.assign(
                self.trf_blocks[b].att.out_projection.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
            )

            self.trf_blocks[b].ff.layers[0].weight = self.assign(
                self.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
            )
            self.trf_blocks[b].ff.layers[0].bias = self.assign(
                self.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
            )
            self.trf_blocks[b].ff.layers[2].weight = self.assign(
                self.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].ff.layers[2].bias = self.assign(
                self.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
            )

            self.trf_blocks[b].norm1.scale = self.assign(
                self.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
            )
            self.trf_blocks[b].norm1.shift = self.assign(
                self.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
            )
            self.trf_blocks[b].norm2.scale = self.assign(
                self.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
            )
            self.trf_blocks[b].norm2.shift = self.assign(
                self.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
            )

        self.final_norm.scale = self.assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = self.assign(self.final_norm.shift, params["b"])
        # 4 The original GPT-2 model by OpenAI reused the token embedding weights in the output layer to reduce the total number
        # of parameters, which is a concept known as weight tying.
        self.out_head.weight = self.assign(self.out_head.weight, params["wte"])  # 4
