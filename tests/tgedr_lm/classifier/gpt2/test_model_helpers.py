"""Focused unit tests for GPT2 model helper branches."""

import numpy as np
import pytest
import torch

from tgedr_lm.classifier.gpt2.model import GPT2Classifier
from tgedr_lm.configuration import ClassifierBaseConfiguration


def _cfg(**overrides) -> ClassifierBaseConfiguration:
    defaults = dict(
        vocabulary_size=64,
        embeddings_dimension=16,
        context_length=8,
        n_layers=1,
        drop_rate=0.0,
        stride=1,
        n_heads=4,
        qkv_bias=True,
        n_classes=3,
    )
    defaults.update(overrides)
    return ClassifierBaseConfiguration(**defaults)


def _make_params(cfg: ClassifierBaseConfiguration) -> dict:
    d_model = cfg.embeddings_dimension
    block = {
        "attn": {
            "c_attn": {"w": np.zeros((d_model, 3 * d_model)), "b": np.zeros(3 * d_model)},
            "c_proj": {"w": np.zeros((d_model, d_model)), "b": np.zeros(d_model)},
        },
        "mlp": {
            "c_fc": {"w": np.zeros((d_model, 4 * d_model)), "b": np.zeros(4 * d_model)},
            "c_proj": {"w": np.zeros((4 * d_model, d_model)), "b": np.zeros(d_model)},
        },
        "ln_1": {"g": np.ones(d_model), "b": np.zeros(d_model)},
        "ln_2": {"g": np.ones(d_model), "b": np.zeros(d_model)},
    }
    return {
        "wte": np.zeros((cfg.vocabulary_size, d_model)),
        "wpe": np.zeros((cfg.context_length, d_model)),
        "blocks": [block] * cfg.n_layers,
        "g": np.ones(d_model),
        "b": np.zeros(d_model),
    }


def test_compute_metrics_handles_tuple_tensor_and_one_hot_labels() -> None:
    logits = torch.tensor(
        [
            [[0.1, 0.1, 0.1], [0.9, 0.1, 0.0]],
            [[0.1, 0.1, 0.1], [0.2, 0.6, 0.2]],
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.6]],
        ]
    )
    labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    metrics = GPT2Classifier.compute_metrics(((logits,), labels))

    assert set(metrics) == {"accuracy", "precision", "recall", "f"}
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_calculate_batch_loss_and_infer_paths() -> None:
    model = GPT2Classifier(_cfg())
    inputs = torch.randint(0, 64, (2, 5))
    labels = torch.randint(0, 3, (2,))

    # device=None path and logger/debug branch lines are exercised by method body.
    loss = model.calculate_batch_loss(inputs, labels)
    assert loss.ndim == 0

    model.train()
    preds = model.infer(inputs)
    assert preds.shape == (2,)
    assert model.training


def test_assign_raises_for_shape_mismatch() -> None:
    model = GPT2Classifier(_cfg())

    with pytest.raises(ValueError, match="Shape mismatch"):
        model._assign(torch.zeros(2, 2), np.zeros((2, 3)))


def test_pretrain_loads_weights_and_toggles_trainable_layers() -> None:
    cfg = _cfg(qkv_bias=True)
    model = GPT2Classifier(cfg)

    model.pretrain(_make_params(cfg))

    for param in model.trf_blocks[-1].parameters():
        assert param.requires_grad
    for param in model.final_norm.parameters():
        assert param.requires_grad
    for param in model.out_head.parameters():
        assert param.requires_grad
