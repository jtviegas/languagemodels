"""Configuration dataclasses and presets for GPT-2 language models."""

from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """Configuration for a base language model.

    Attributes
    ----------
    vocabulary_size : int
        Number of tokens in the vocabulary.
    embeddings_dimension : int
        Dimension of token and positional embeddings.
    context_length : int
        Maximum number of tokens in a sequence.
    n_layers : int
        Number of transformer blocks.
    drop_rate : float
        Dropout probability for regularization.
    stride : int
        Stride used for sliding window tokenization.
    n_heads : int
        Number of attention heads per transformer block.
    qkv_bias : bool
        Whether to use bias in query, key, and value projections.
    """

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
