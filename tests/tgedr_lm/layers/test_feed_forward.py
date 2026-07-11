"""Unit tests for the feed_forward layer module."""

import pytest
import torch
from tgedr_lm.layers.feed_forward import FeedForward
from tgedr_lm.configuration import BaseModelConfig


class TestFeedForward:
    """Test suite for FeedForward layer."""

    def get_test_config(self) -> BaseModelConfig:
        """Get a test configuration."""
        return BaseModelConfig(
            vocabulary_size=50257,
            embeddings_dimension=128,
            context_length=512,
            n_layers=2,
            drop_rate=0.1,
            stride=1,
            n_heads=8,
        )

    def test_feedforward_initialization(self) -> None:
        """Test FeedForward initialization."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        assert ff.layers is not None
        assert len(ff.layers) == 3  # Linear, GELU, Linear

    def test_feedforward_forward_shape(self) -> None:
        """Test FeedForward forward pass shape."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, cfg.embeddings_dimension)
        output = ff(x)
        
        assert output.shape == x.shape

    def test_feedforward_expansion_factor(self) -> None:
        """Test that first layer expands by 4x."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        # First linear layer should expand by 4x
        assert ff.layers[0].out_features == 4 * cfg.embeddings_dimension
        # Third layer should project back to embedding dimension
        assert ff.layers[2].out_features == cfg.embeddings_dimension

    def test_feedforward_different_embedding_dims(self) -> None:
        """Test FeedForward with different embedding dimensions."""
        for emb_dim in [64, 128, 256, 512]:
            cfg = BaseModelConfig(
                vocabulary_size=50257,
                embeddings_dimension=emb_dim,
                context_length=512,
                n_layers=2,
                drop_rate=0.1,
                stride=1,
                n_heads=8,
            )
            ff = FeedForward(cfg)
            
            x = torch.randn(2, 5, emb_dim)
            output = ff(x)
            assert output.shape == (2, 5, emb_dim)

    def test_feedforward_batch_sizes(self) -> None:
        """Test FeedForward with different batch sizes."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, cfg.embeddings_dimension)
            output = ff(x)
            assert output.shape == (batch_size, 10, cfg.embeddings_dimension)

    def test_feedforward_sequence_lengths(self) -> None:
        """Test FeedForward with different sequence lengths."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        for seq_len in [1, 5, 10, 50]:
            x = torch.randn(2, seq_len, cfg.embeddings_dimension)
            output = ff(x)
            assert output.shape == (2, seq_len, cfg.embeddings_dimension)

    def test_feedforward_gradient_flow(self) -> None:
        """Test that gradients flow through FeedForward."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        x = torch.randn(2, 10, cfg.embeddings_dimension, requires_grad=True)
        output = ff(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that all layers have gradients
        assert ff.layers[0].weight.grad is not None
        assert ff.layers[2].weight.grad is not None

    def test_feedforward_parameters(self) -> None:
        """Test that FeedForward has learnable parameters."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        params = list(ff.parameters())
        assert len(params) > 0

    def test_feedforward_contains_gelu(self) -> None:
        """Test that FeedForward contains GELU activation."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        from tgedr_lm.activations import GELU
        gelu_found = any(isinstance(layer, GELU) for layer in ff.layers)
        assert gelu_found

    def test_feedforward_output_dtype(self) -> None:
        """Test that output has same dtype as input."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        
        x = torch.randn(2, 10, cfg.embeddings_dimension, dtype=torch.float32)
        output = ff(x)
        assert output.dtype == x.dtype

    def test_feedforward_eval_mode(self) -> None:
        """Test FeedForward in eval mode."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        ff.eval()
        
        x = torch.randn(2, 10, cfg.embeddings_dimension)
        output = ff(x)
        assert output.shape == x.shape

    def test_feedforward_train_mode(self) -> None:
        """Test FeedForward in train mode."""
        cfg = self.get_test_config()
        ff = FeedForward(cfg)
        ff.train()
        
        x = torch.randn(2, 10, cfg.embeddings_dimension)
        output = ff(x)
        assert output.shape == x.shape
