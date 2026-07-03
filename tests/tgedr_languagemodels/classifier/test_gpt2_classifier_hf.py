"""Unit tests for the Hugging Face Trainer-compatible GPT2Classifier."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tgedr_languagemodels.classifier.gpt2.configuration import ClassifierConfiguration
from tgedr_languagemodels.classifier.gpt2.model import GPT2Classifier
from transformers.modeling_outputs import SequenceClassifierOutput


def _cfg(**overrides) -> ClassifierConfiguration:
    defaults = dict(
        vocabulary_size=64,
        embeddings_dimension=16,
        context_length=8,
        n_layers=1,
        drop_rate=0.0,
        stride=1,
        n_heads=4,
        qkv_bias=False,
        n_classes=3,
    )
    defaults.update(overrides)
    return ClassifierConfiguration(**defaults)


class TestGPT2ClassifierInit:

    def test_layers_created(self) -> None:
        model = GPT2Classifier(_cfg())
        assert model.tok_emb is not None
        assert model.pos_emb is not None
        assert model.drop_emb is not None
        assert model.trf_blocks is not None
        assert model.final_norm is not None
        assert model.out_head is not None

    def test_embedding_dimensions(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        assert model.tok_emb.num_embeddings == cfg.vocabulary_size
        assert model.tok_emb.embedding_dim == cfg.embeddings_dimension
        assert model.pos_emb.num_embeddings == cfg.context_length
        assert model.pos_emb.embedding_dim == cfg.embeddings_dimension

    def test_output_head_shape(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        assert model.out_head.out_features == cfg.n_classes

    def test_no_custom_training_flag(self) -> None:
        model = GPT2Classifier(_cfg())
        assert not hasattr(model, "_training") or isinstance(model._training, bool) is False


class TestGPT2ClassifierForward:

    def test_returns_sequence_classifier_output(self) -> None:
        model = GPT2Classifier(_cfg())
        inputs = torch.randint(0, 64, (2, 5))
        out = model(input_ids=inputs)
        assert isinstance(out, SequenceClassifierOutput)

    def test_logits_shape(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        batch, seq = 2, 5
        inputs = torch.randint(0, cfg.vocabulary_size, (batch, seq))
        out = model(input_ids=inputs)
        assert out.logits.shape == (batch, seq, cfg.n_classes)

    def test_no_loss_without_labels(self) -> None:
        model = GPT2Classifier(_cfg())
        inputs = torch.randint(0, 64, (2, 5))
        out = model(input_ids=inputs)
        assert out.loss is None

    def test_loss_present_with_labels(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (2, 5))
        labels = torch.randint(0, cfg.n_classes, (2,))
        out = model(input_ids=inputs, labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0

    def test_loss_is_scalar_tensor(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (3, 6))
        labels = torch.randint(0, cfg.n_classes, (3,))
        out = model(input_ids=inputs, labels=labels)
        assert isinstance(out.loss, torch.Tensor)
        assert out.loss.shape == torch.Size([])

    def test_attention_mask_and_extra_kwargs_ignored(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (2, 5))
        mask = torch.ones(2, 5)
        # Should not raise
        out = model(input_ids=inputs, attention_mask=mask, token_type_ids=None)
        assert out.logits is not None


class TestGPT2ClassifierCalculateBatchLoss:

    def test_returns_scalar(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (4, 5))
        labels = torch.randint(0, cfg.n_classes, (4,))
        loss = model.calculate_batch_loss(inputs, labels)
        assert loss.ndim == 0

    def test_loss_is_non_negative(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (4, 5))
        labels = torch.randint(0, cfg.n_classes, (4,))
        loss = model.calculate_batch_loss(inputs, labels)
        assert loss.item() >= 0.0

    def test_accepts_device_kwarg(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (2, 5))
        labels = torch.randint(0, cfg.n_classes, (2,))
        loss = model.calculate_batch_loss(inputs, labels, device=torch.device("cpu"))
        assert loss.ndim == 0

    def test_consistent_with_forward_loss(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        model.eval()
        torch.manual_seed(0)
        inputs = torch.randint(0, cfg.vocabulary_size, (2, 5))
        labels = torch.randint(0, cfg.n_classes, (2,))
        with torch.no_grad():
            fwd_loss = model(input_ids=inputs, labels=labels).loss
            batch_loss = model.calculate_batch_loss(inputs, labels)
        assert torch.isclose(fwd_loss, batch_loss, atol=1e-5)


class TestGPT2ClassifierInfer:

    def test_returns_predicted_labels(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (3, 5))
        preds = model.infer(inputs)
        assert preds.shape == (3,)

    def test_predictions_within_class_range(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        inputs = torch.randint(0, cfg.vocabulary_size, (4, 5))
        preds = model.infer(inputs)
        assert preds.min().item() >= 0
        assert preds.max().item() < cfg.n_classes

    def test_infer_restores_training_mode(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        model.train()
        assert model.training
        model.infer(torch.randint(0, cfg.vocabulary_size, (2, 5)))
        assert model.training

    def test_infer_stays_eval_if_already_eval(self) -> None:
        cfg = _cfg()
        model = GPT2Classifier(cfg)
        model.eval()
        model.infer(torch.randint(0, cfg.vocabulary_size, (2, 5)))
        assert not model.training


class TestGPT2ClassifierPretrain:

    def _make_params(self, cfg: ClassifierConfiguration) -> dict:
        D, C, H = cfg.embeddings_dimension, cfg.context_length, cfg.n_heads
        head_dim = D // H
        block = {
            "attn": {
                "c_attn": {"w": np.zeros((D, 3 * D)), "b": np.zeros(3 * D)},
                "c_proj": {"w": np.zeros((D, D)), "b": np.zeros(D)},
            },
            "mlp": {
                "c_fc":   {"w": np.zeros((D, 4 * D)), "b": np.zeros(4 * D)},
                "c_proj": {"w": np.zeros((4 * D, D)), "b": np.zeros(D)},
            },
            "ln_1": {"g": np.ones(D), "b": np.zeros(D)},
            "ln_2": {"g": np.ones(D), "b": np.zeros(D)},
        }
        return {
            "wte": np.zeros((cfg.vocabulary_size, D)),
            "wpe": np.zeros((C, D)),
            "blocks": [block] * cfg.n_layers,
            "g": np.ones(D),
            "b": np.zeros(D),
        }

    def test_pretrain_freezes_most_params(self) -> None:
        cfg = _cfg(qkv_bias=True)
        model = GPT2Classifier(cfg)
        model.pretrain(self._make_params(cfg))
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(trainable) > 0
        assert len(frozen) > 0

    def test_pretrain_keeps_last_block_trainable(self) -> None:
        cfg = _cfg(qkv_bias=True)
        model = GPT2Classifier(cfg)
        model.pretrain(self._make_params(cfg))
        for p in model.trf_blocks[-1].parameters():
            assert p.requires_grad

    def test_pretrain_keeps_out_head_trainable(self) -> None:
        cfg = _cfg(qkv_bias=True)
        model = GPT2Classifier(cfg)
        model.pretrain(self._make_params(cfg))
        for p in model.out_head.parameters():
            assert p.requires_grad

    def test_pretrain_keeps_final_norm_trainable(self) -> None:
        cfg = _cfg(qkv_bias=True)
        model = GPT2Classifier(cfg)
        model.pretrain(self._make_params(cfg))
        for p in model.final_norm.parameters():
            assert p.requires_grad
