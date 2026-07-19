"""Unit tests for the activations module."""

import pytest
import torch
from tgedr_lm.activations import GELU


class TestGELU:
    """Test suite for the GELU activation function."""

    def test_gelu_initialization(self) -> None:
        """Test that GELU can be instantiated."""
        gelu = GELU()
        assert isinstance(gelu, torch.nn.Module)

    def test_gelu_forward_single_value(self) -> None:
        """Test GELU forward pass with a single value."""
        gelu = GELU()
        x = torch.tensor([[0.0]])
        output = gelu(x)
        assert output.shape == x.shape
        # At x=0, GELU should output 0
        assert torch.allclose(output, torch.tensor([[0.0]]), atol=1e-6)

    def test_gelu_forward_positive(self) -> None:
        """Test GELU forward pass with positive values."""
        gelu = GELU()
        x = torch.tensor([1.0, 2.0, 3.0])
        output = gelu(x)
        assert output.shape == x.shape
        # GELU should output positive values for positive inputs
        assert torch.all(output > 0)
        # GELU should be less than x for large positive values (smooth approximation)
        assert torch.all(output[1:] < x[1:])

    def test_gelu_forward_negative(self) -> None:
        """Test GELU forward pass with negative values."""
        gelu = GELU()
        x = torch.tensor([-1.0, -2.0, -3.0])
        output = gelu(x)
        assert output.shape == x.shape
        # GELU allows small negative values through
        assert torch.all(output < 0)

    def test_gelu_forward_batch(self) -> None:
        """Test GELU forward pass with batch input."""
        gelu = GELU()
        x = torch.randn(4, 8, 16)
        output = gelu(x)
        assert output.shape == x.shape

    def test_gelu_gradient_flow(self) -> None:
        """Test that gradients flow through GELU."""
        gelu = GELU()
        x = torch.randn(2, 3, requires_grad=True)
        output = gelu(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gelu_differentiable(self) -> None:
        """Test that GELU is differentiable."""
        gelu = GELU().double()
        torch.manual_seed(0)
        # Keep values moderate to avoid finite-difference instability in tanh-based GELU approximation.
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True) * 0.5
        assert torch.autograd.gradcheck(gelu, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gelu_antisymmetric_tendency(self) -> None:
        """Test GELU has some antisymmetric-like behavior."""
        gelu = GELU()
        x_pos = torch.tensor([1.0])
        x_neg = torch.tensor([-1.0])
        out_pos = gelu(x_pos)
        out_neg = gelu(x_neg)
        # GELU should have opposite signs for opposite inputs
        assert torch.sign(out_pos) != torch.sign(out_neg) or (out_pos.abs() < 0.01)
