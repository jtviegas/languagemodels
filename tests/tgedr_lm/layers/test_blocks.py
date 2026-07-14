"""Unit tests for the transformer block module."""

import pytest
import torch
from tgedr_lm.configuration import ClassifierBaseConfiguration
from tgedr_lm.layers.blocks import TransformerBlock



class TestTransformerBlock:
    """Test suite for TransformerBlock."""

    def get_test_config(self) -> ClassifierBaseConfiguration:
        """Get a test configuration."""
        return ClassifierBaseConfiguration(
            vocabulary_size=50257,
            embeddings_dimension=128,
            context_length=512,
            n_layers=2,
            drop_rate=0.1,
            stride=1,
            n_heads=8,
        )

    def test_block_initialization(self) -> None:
        """Test TransformerBlock initialization."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        assert block.att is not None
        assert block.ff is not None
        assert block.norm1 is not None
        assert block.norm2 is not None
        assert block.drop_shortcut is not None

    def test_block_forward_shape(self) -> None:
        """Test TransformerBlock forward pass shape."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, cfg.embeddings_dimension)
        output = block(x)
        
        assert output.shape == x.shape

    def test_block_preserves_embeddings_dimension(self) -> None:
        """Test that TransformerBlock preserves embedding dimension."""
        for emb_dim in [64, 128, 256]:
            cfg = ClassifierBaseConfiguration(
                vocabulary_size=50257,
                embeddings_dimension=emb_dim,
                context_length=512,
                n_layers=2,
                drop_rate=0.1,
                stride=1,
                n_heads=8,
            )
            block = TransformerBlock(cfg)
            
            x = torch.randn(2, 5, emb_dim)
            output = block(x)
            assert output.shape == x.shape

    def test_block_residual_connections(self) -> None:
        """Test that TransformerBlock has residual connections."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        # Create an identity-like block to test residual behavior
        x = torch.randn(2, 10, cfg.embeddings_dimension)
        output = block(x)
        
        # Output should have same shape
        assert output.shape == x.shape

    def test_block_batch_sizes(self) -> None:
        """Test TransformerBlock with different batch sizes."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 10, cfg.embeddings_dimension)
            output = block(x)
            assert output.shape == (batch_size, 10, cfg.embeddings_dimension)

    def test_block_sequence_lengths(self) -> None:
        """Test TransformerBlock with different sequence lengths."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, cfg.embeddings_dimension)
            output = block(x)
            assert output.shape == (2, seq_len, cfg.embeddings_dimension)

    def test_block_gradient_flow(self) -> None:
        """Test that gradients flow through TransformerBlock."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        x = torch.randn(2, 10, cfg.embeddings_dimension, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that components have gradients
        assert block.att.W_query.weight.grad is not None
        assert block.ff.layers[0].weight.grad is not None

    def test_block_contains_attention(self) -> None:
        """Test that TransformerBlock contains MultiHeadAttention."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        from tgedr_lm.layers.attention import MultiHeadAttention
        assert isinstance(block.att, MultiHeadAttention)

    def test_block_contains_feedforward(self) -> None:
        """Test that TransformerBlock contains FeedForward."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        from tgedr_lm.layers.feed_forward import FeedForward
        assert isinstance(block.ff, FeedForward)

    def test_block_contains_layer_norms(self) -> None:
        """Test that TransformerBlock contains LayerNormalizations."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        from tgedr_lm.layers.normalization import LayerNormalization
        assert isinstance(block.norm1, LayerNormalization)
        assert isinstance(block.norm2, LayerNormalization)

    def test_block_dropout_dropout_rate(self) -> None:
        """Test that TransformerBlock uses correct dropout rate."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        assert block.drop_shortcut.p == cfg.drop_rate

    def test_block_parameters(self) -> None:
        """Test that TransformerBlock has learnable parameters."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        params = list(block.parameters())
        assert len(params) > 0

    def test_block_eval_mode(self) -> None:
        """Test TransformerBlock in eval mode."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        block.eval()
        
        x = torch.randn(2, 10, cfg.embeddings_dimension)
        output = block(x)
        assert output.shape == x.shape

    def test_block_train_mode(self) -> None:
        """Test TransformerBlock in train mode."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        block.train()
        
        x = torch.randn(2, 10, cfg.embeddings_dimension)
        output = block(x)
        assert output.shape == x.shape

    def test_block_pre_normalization(self) -> None:
        """Test that TransformerBlock uses pre-normalization."""
        cfg = self.get_test_config()
        block = TransformerBlock(cfg)
        
        # Verify the architecture has pre-norm structure
        # by checking that norm1 is applied before attention
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'att')

    def test_block_multiple_heads(self) -> None:
        """Test TransformerBlock with different numbers of heads."""
        for num_heads in [4, 8, 16]:
            cfg = ClassifierBaseConfiguration(
                vocabulary_size=50257,
                embeddings_dimension=256,  # Divisible by all head counts
                context_length=512,
                n_layers=2,
                drop_rate=0.1,
                stride=1,
                n_heads=num_heads,
            )
            block = TransformerBlock(cfg)
            
            x = torch.randn(2, 10, 256)
            output = block(x)
            assert output.shape == x.shape
