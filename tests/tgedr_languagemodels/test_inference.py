"""Unit tests for the inference module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
from tgedr_languagemodels.inference import generate


class DummyModel(nn.Module):
    """Simple model for testing text generation."""

    def __init__(self, vocab_size: int = 100) -> None:
        """Initialize dummy model."""
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dummy logits."""
        batch_size, seq_len = x.shape
        # Return uniform logits
        logits = torch.ones(batch_size, seq_len, self.vocab_size)
        return logits


class EosModel(nn.Module):
    """Model that forces EOS token at each step."""

    def __init__(self, vocab_size: int, eos_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[:, :, self.eos_id] = 100.0
        return logits


class TestGenerate:
    """Test suite for the generate function."""

    def test_generate_basic(self) -> None:
        """Test basic text generation."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        context_size = 10
        max_new_tokens = 5
        
        result = idx
        for _ in range(max_new_tokens):
            idx_cond = result[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            result = torch.cat((result, idx_next), dim=1)
        
        # Verify output length
        assert result.shape[1] == 3 + max_new_tokens

    def test_generate_greedy_selection(self) -> None:
        """Test generation with greedy selection (temperature=0)."""
        model = DummyModel(vocab_size=10)
        idx = torch.tensor([[0]])
        
        result = generate(
            model, 
            idx, 
            max_new_tokens=5, 
            context_size=10,
            temperature=0.0
        )
        
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == 6  # original + 5 generated tokens
        assert torch.all(result < 10)  # all tokens should be valid

    def test_generate_with_eos_token(self) -> None:
        """Test generation with EOS token."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        
        # Use an impossible EOS token so generation continues
        result = generate(
            model,
            idx,
            max_new_tokens=5,
            context_size=10,
            temperature=0.0,
            eos_id=999  # Token ID that won't be generated
        )
        
        # Result should have generated new tokens
        assert result.shape[1] == 3 + 5

    def test_generate_with_top_k(self) -> None:
        """Test generation with top-k sampling."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        
        result = generate(
            model,
            idx,
            max_new_tokens=5,
            context_size=10,
            temperature=1.0,
            top_k=10
        )
        
        assert result.shape[0] == 1
        assert result.shape[1] == 3 + 5

    def test_generate_with_temperature(self) -> None:
        """Test generation with temperature scaling."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        
        result = generate(
            model,
            idx,
            max_new_tokens=5,
            context_size=10,
            temperature=0.5
        )
        
        assert result.shape[0] == 1
        assert result.shape[1] == 3 + 5

    def test_generate_batch_input(self) -> None:
        """Test generation with batch input."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2], [3, 4], [5, 6]])
        
        result = generate(
            model,
            idx,
            max_new_tokens=3,
            context_size=10,
            temperature=0.0
        )
        
        assert result.shape[0] == 3  # batch size
        assert result.shape[1] == 2 + 3  # original + generated

    def test_generate_preserves_dtype(self) -> None:
        """Test that generation preserves input dtype."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        result = generate(
            model,
            idx,
            max_new_tokens=2,
            context_size=10,
            temperature=0.0
        )
        
        assert result.dtype == torch.long

    def test_generate_context_window(self) -> None:
        """Test that generation respects context window."""
        model = DummyModel(vocab_size=100)
        idx = torch.arange(20).unsqueeze(0)  # sequence of 20 tokens
        context_size = 5
        
        # Generate with small context window
        result = generate(
            model,
            idx,
            max_new_tokens=3,
            context_size=context_size,
            temperature=0.0
        )
        
        assert result.shape[1] == 20 + 3

    def test_generate_max_tokens_zero(self) -> None:
        """Test generation with zero max_new_tokens."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        
        result = generate(
            model,
            idx,
            max_new_tokens=0,
            context_size=10,
            temperature=0.0
        )
        
        assert torch.equal(result, idx)

    def test_generate_top_k_and_temperature(self) -> None:
        """Test generation with both top-k and temperature."""
        model = DummyModel(vocab_size=100)
        idx = torch.tensor([[1, 2, 3]])
        
        result = generate(
            model,
            idx,
            max_new_tokens=5,
            context_size=10,
            temperature=0.8,
            top_k=20
        )
        
        assert result.shape[0] == 1
        assert result.shape[1] == 3 + 5

    def test_generate_breaks_early_on_eos(self) -> None:
        """Generation should stop before appending when EOS is predicted."""
        eos_id = 2
        model = EosModel(vocab_size=10, eos_id=eos_id)
        idx = torch.tensor([[1, 4, 5]])

        result = generate(
            model,
            idx,
            max_new_tokens=5,
            context_size=10,
            temperature=0.0,
            eos_id=eos_id,
        )

        assert torch.equal(result, idx)
