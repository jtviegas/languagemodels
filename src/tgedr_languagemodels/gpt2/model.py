"""GPT model definition and supporting layers for language modeling."""

from attr import dataclass

from tgedr_languagemodels.blocks import TransformerBlock
from tgedr_languagemodels.layers import LayerNorm
import torch
from torch import nn


@dataclass
class ModelConfig:
    vocabulary_size: int
    embeddings_dimension: int
    sequence_length: int
    n_layers: int
    drop_rate: float

class GPT2Model(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
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
