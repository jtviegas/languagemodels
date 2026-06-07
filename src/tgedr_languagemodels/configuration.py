from dataclasses import dataclass
from typing import ClassVar


@dataclass
class BaseModelConfig:
    vocabulary_size: int
    embeddings_dimension: int
    context_length: int
    n_layers: int
    drop_rate: float
    stride: int
    n_heads: int
    qkv_bias: bool = False

GPT2_MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
