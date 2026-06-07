
import torch
from torch import nn

from tgedr_languagemodels.activations import GELU
from tgedr_languagemodels.configuration import BaseModelConfig



class FeedForward(nn.Module):
    """Transformer feed-forward block with expansion and projection layers."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embeddings_dimension, 4 * cfg.embeddings_dimension),
            GELU(),
            nn.Linear(4 * cfg.embeddings_dimension, cfg.embeddings_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
