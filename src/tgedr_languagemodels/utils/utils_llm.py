"""Utility functions for language model text generation.

This module provides text generation utilities including top-k sampling,
temperature scaling, and greedy token selection for transformer models.
"""

import torch
import json
import gzip
import pickle  # nosec B403 - Used for trusted internal serialization, not untrusted input
from pathlib import Path
import pandas as pd


def text_to_token_ids(text, tokenizer) -> torch.Tensor:
    """Encode a text string into a batched token-id tensor.

    Parameters
    ----------
    text : str
        Input text to encode.
    tokenizer : tiktoken.Encoding
        Tokenizer used for encoding.

    Returns
    -------
    torch.Tensor
        Token-id tensor of shape (1, seq_len).
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 1  .unsqueeze(0) adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer) -> str:
    """Decode a batched token-id tensor back into a text string.

    Parameters
    ----------
    token_ids : torch.Tensor
        Token-id tensor of shape (1, seq_len).
    tokenizer : tiktoken.Encoding
        Tokenizer used for decoding.

    Returns
    -------
    str
        Decoded text string.
    """
    flat = token_ids.squeeze(0)  # 2 Removes batch dimension
    return tokenizer.decode(flat.tolist())


# ### ===>>> Top-k sampling - when combined with probabilistic sampling and temperature scaling, can improve the text generation results.
# In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from the selection process
# by masking their probability scores
# The top-k approach replaces all nonselected logits with negative infinity value (-inf), such that when computing the softmax values,
# the probability scores of the non-top-k tokens are 0, and the remaining probabilities sum up to 1


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None) -> torch.Tensor:
    """Generate new tokens autoregressively using optional top-k sampling and temperature scaling.

    Parameters
    ----------
    model : nn.Module
        Language model that returns logits of shape (batch, seq_len, vocab_size).
    idx : torch.Tensor
        Starting token-id context of shape (batch, seq_len).
    max_new_tokens : int
        Maximum number of tokens to generate.
    context_size : int
        Maximum context window the model can handle.
    temperature : float, optional
        Scaling factor for logits; 0.0 uses greedy selection (default: 0.0).
    top_k : int or None, optional
        If set, restricts sampling to the top-k most likely tokens (default: None).
    eos_id : int or None, optional
        Token id at which generation stops early (default: None).

    Returns
    -------
    torch.Tensor
        Token-id tensor of shape (batch, seq_len + n_generated).
    """
    for _ in range(max_new_tokens):  # 1
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:  # 2
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        if temperature > 0.0:  # 3
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # 4
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:  # 5
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# 1 The for loop is the same as before: gets logits and only focuses on the last time step.
# 2 Filters logits with top_k sampling
# 3 Applies temperature scaling
# 4 Carries out greedy next-token selection as before when temperature scaling is disabled
# 5 Stops generating early if end-of-sequence token is encountered


def _is_pickle_path(path: Path) -> bool:
    """Return True if the path has a pickle or compressed-pickle suffix.

    Parameters
    ----------
    path : Path
        File path to inspect.

    Returns
    -------
    bool
        True when the path ends in .pickle, .pkl, or .pickle.gz / .pkl.gz.
    """
    suffixes = path.suffixes
    if suffixes[-1] == ".gz" and len(suffixes) > 1:
        return suffixes[-2] in {".pickle", ".pkl"}
    return suffixes[-1] in {".pickle", ".pkl"}


def save_dict(data, target_path, *, use_pickle: bool | None = None, compress: bool = False) -> None:
    """Serialize a dictionary to disk as JSON or pickle, with optional gzip compression.

    Parameters
    ----------
    data : dict
        Data to serialize.
    target_path : str or Path
        Destination file path.
    use_pickle : bool or None, optional
        Force pickle format; inferred from file extension when None (default: None).
    compress : bool, optional
        Whether to apply gzip compression (default: False).
    """
    path = Path(target_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dict(source_path, *, use_pickle: bool | None = None, compress: bool = False) -> dict:
    """Deserialize a dictionary from a JSON or pickle file, with optional gzip decompression.

    Parameters
    ----------
    source_path : str or Path
        Source file path.
    use_pickle : bool or None, optional
        Force pickle format; inferred from file extension when None (default: None).
    compress : bool, optional
        Whether the file is gzip-compressed (default: False).

    Returns
    -------
    dict
        Deserialized data.
    """
    path = Path(source_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 # nosec B301

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_pickle_compressed(data, target_path) -> None:
    """Save Python data using pickle and gzip compression."""
    save_dict(data, target_path, use_pickle=True, compress=True)


def load_pickle_compressed(source_path) -> dict:
    """Load Python data saved with pickle and gzip compression."""
    return load_dict(source_path, use_pickle=True, compress=True)


def longest_encoded_length(encoded_texts: list[list[int]]) -> int:
    """Return the length of the longest encoded text in the list.

    Parameters
    ----------
    encoded_texts : list[list[int]]
        List of tokenized texts (each as a list of integer token ids).

    Returns
    -------
    int
        Length of the longest token sequence.
    """
    max_length = 0
    for encoded_text in encoded_texts:
        max_length = max(max_length, len(encoded_text))
    return max_length


def harmonize_text_sequences(df: pd.DataFrame, tokenizer, text_col="text", sequence_length=None) -> list[list[int]]:
    """Encode and pad text sequences to a uniform length.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the text column.
    tokenizer : tiktoken.Encoding
        Tokenizer used for encoding.
    text_col : str, optional
        Name of the column with text data (default: "text").
    sequence_length : int or None, optional
        Target sequence length; uses the longest encoded text when None (default: None).

    Returns
    -------
    list[list[int]]
        List of token-id sequences padded to the same length with the EOT token.
    """
    encoded_texts: list[list[int]] = [tokenizer.encode(text) for text in df[text_col]]
    max_length = longest_encoded_length(encoded_texts) if sequence_length is None else sequence_length
    encoded_texts = [
        encoded_text + [tokenizer.eot_token] * (max_length - len(encoded_text)) for encoded_text in encoded_texts
    ]
    return encoded_texts
