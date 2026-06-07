import torch
import torch.nn as nn

from tgedr_languagemodels.configuration import BaseModelConfig
from tgedr_languagemodels.layers.attention import MultiHeadAttention
from tgedr_languagemodels.layers.feed_forward import FeedForward
from tgedr_languagemodels.layers.normalization import LayerNormalization


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BaseModelConfig):
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
