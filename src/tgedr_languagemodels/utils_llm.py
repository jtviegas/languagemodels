"""Utility functions for language model text generation.

This module provides text generation utilities including top-k sampling,
temperature scaling, and greedy token selection for transformer models.
"""

import torch

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
