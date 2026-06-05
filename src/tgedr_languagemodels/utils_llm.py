"""Utility functions for language model text generation.

This module provides text generation utilities including top-k sampling,
temperature scaling, and greedy token selection for transformer models.
"""

import torch
import json
import gzip
import pickle
from pathlib import Path
import pandas as pd


def text_to_token_ids(text, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 1  .unsqueeze(0) adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer) -> str:
    flat = token_ids.squeeze(0)  # 2 Removes batch dimension
    return tokenizer.decode(flat.tolist())


# ### ===>>> Top-k sampling - when combined with probabilistic sampling and temperature scaling, can improve the text generation results.
# In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from the selection process
# by masking their probability scores
# The top-k approach replaces all nonselected logits with negative infinity value (-inf), such that when computing the softmax values,
# the probability scores of the non-top-k tokens are 0, and the remaining probabilities sum up to 1


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
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
    suffixes = path.suffixes
    if suffixes[-1] == ".gz" and len(suffixes) > 1:
        return suffixes[-2] in {".pickle", ".pkl"}
    return suffixes[-1] in {".pickle", ".pkl"}


def save_dict(data, target_path, *, use_pickle: bool | None = None, compress: bool = False) -> None:
    path = Path(target_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dict(source_path, *, use_pickle: bool | None = None, compress: bool = False):
    path = Path(source_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            return pickle.load(f)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle_compressed(data, target_path) -> None:
    """Save Python data using pickle and gzip compression."""
    save_dict(data, target_path, use_pickle=True, compress=True)


def load_pickle_compressed(source_path):
    """Load Python data saved with pickle and gzip compression."""
    return load_dict(source_path, use_pickle=True, compress=True)

def longest_encoded_length(encoded_texts: list[list[int]]) -> int:
    max_length = 0
    for encoded_text in encoded_texts:
        encoded_length = len(encoded_text)
        if encoded_length > max_length:
            max_length = encoded_length
    return max_length

def harmonize_text_sequences(df: pd.DataFrame, tokenizer, text_col="text"):
    encoded_texts: list[list[int]] = [tokenizer.encode(text) for text in df[text_col]]
    max_length = longest_encoded_length(encoded_texts)
    encoded_texts = [
        encoded_text + [tokenizer.eot_token] * 
        (max_length - len(encoded_text))
        for encoded_text in encoded_texts
    ]
    return encoded_texts
