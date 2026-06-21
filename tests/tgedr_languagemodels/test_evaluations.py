"""Unit tests for evaluation mixins."""

import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tgedr_languagemodels.evaluations import CrossEntropyModelEvaluatorMixin


class DummyEvalModel(CrossEntropyModelEvaluatorMixin, nn.Module):
    """Minimal model to exercise evaluation mixin behavior."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, self.n_classes, dtype=torch.float32)
        cls = (x[:, 0] % self.n_classes).long()
        logits[torch.arange(batch_size), -1, cls] = 5.0
        return logits


class TestCrossEntropyModelEvaluatorMixin:
    """Test suite for CrossEntropyModelEvaluatorMixin."""

    def test_calculate_batch_loss_returns_scalar_tensor(self) -> None:
        model = DummyEvalModel(n_classes=3)
        inputs = torch.tensor([[0, 1], [1, 2], [2, 1]], dtype=torch.long)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)

        loss = model.calculate_batch_loss(inputs, targets, device=torch.device("cpu"))

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_calculate_loader_accuracy_restores_training_mode(self) -> None:
        model = DummyEvalModel(n_classes=3)
        model.train()

        inputs = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 4]], dtype=torch.long)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)

        acc = model.calculate_loader_accuracy(loader, device=torch.device("cpu"), num_batches=None)

        assert model.training is True
        assert acc == 1.0

    def test_calculate_loader_accuracy_keeps_eval_mode(self) -> None:
        model = DummyEvalModel(n_classes=3)
        model.eval()

        inputs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        targets = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)

        _ = model.calculate_loader_accuracy(loader, device=torch.device("cpu"), num_batches=1)

        assert model.training is False

    def test_calculate_loader_accuracy_empty_loader_returns_nan(self) -> None:
        model = DummyEvalModel(n_classes=3)
        model.train()

        empty_inputs = torch.empty((0, 2), dtype=torch.long)
        empty_targets = torch.empty((0,), dtype=torch.long)
        loader = DataLoader(TensorDataset(empty_inputs, empty_targets), batch_size=2, shuffle=False)

        acc = model.calculate_loader_accuracy(loader, device=torch.device("cpu"), num_batches=None)

        assert model.training is True
        assert math.isnan(acc)
