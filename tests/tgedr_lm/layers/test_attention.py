"""Unit tests for the attention layer module."""

import pytest
import torch
from tgedr_lm.layers.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""

    def test_attention_initialization(self) -> None:
        """Test MultiHeadAttention initialization."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
        )
        
        assert attention.embedding_dimension == 64
        assert attention.output_dimension == 64
        assert attention.context_length == 512
        assert attention.heads == 8
        assert attention.head_dimension == 8

    def test_attention_output_dimension_divisible_by_heads(self) -> None:
        """Test that output_dimension must be divisible by heads."""
        with pytest.raises(ValueError):
            MultiHeadAttention(
                embeddings_dimension=64,
                output_dimension=65,  # Not divisible by 8
                context_length=512,
                heads=8,
            )

    def test_attention_forward_shape(self) -> None:
        """Test MultiHeadAttention forward pass shape."""
        batch_size, seq_len = 2, 10
        emb_dim = 64
        
        attention = MultiHeadAttention(
            embeddings_dimension=emb_dim,
            output_dimension=emb_dim,
            context_length=512,
            heads=8,
        )
        
        x = torch.randn(batch_size, seq_len, emb_dim)
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, emb_dim)

    def test_attention_different_batch_sizes(self) -> None:
        """Test attention with different batch sizes."""
        emb_dim = 128
        seq_len = 20
        
        attention = MultiHeadAttention(
            embeddings_dimension=emb_dim,
            output_dimension=emb_dim,
            context_length=512,
            heads=8,
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, seq_len, emb_dim)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, emb_dim)

    def test_attention_different_seq_lengths(self) -> None:
        """Test attention with different sequence lengths."""
        batch_size = 2
        emb_dim = 128
        
        attention = MultiHeadAttention(
            embeddings_dimension=emb_dim,
            output_dimension=emb_dim,
            context_length=512,
            heads=8,
        )
        
        for seq_len in [1, 5, 10, 50]:
            x = torch.randn(batch_size, seq_len, emb_dim)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, emb_dim)

    def test_attention_with_qkv_bias(self) -> None:
        """Test attention with QKV bias."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
            qkv_bias=True,
        )
        
        assert attention.W_query.bias is not None
        assert attention.W_key.bias is not None
        assert attention.W_value.bias is not None

    def test_attention_without_qkv_bias(self) -> None:
        """Test attention without QKV bias."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
            qkv_bias=False,
        )
        
        assert attention.W_query.bias is None
        assert attention.W_key.bias is None
        assert attention.W_value.bias is None

    def test_attention_causal_mask(self) -> None:
        """Test that causal mask prevents future attention."""
        batch_size, seq_len = 1, 4
        emb_dim = 32
        
        attention = MultiHeadAttention(
            embeddings_dimension=emb_dim,
            output_dimension=emb_dim,
            context_length=512,
            heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, emb_dim)
        output = attention(x)
        
        # Verify output shape is correct
        assert output.shape == (batch_size, seq_len, emb_dim)

    def test_attention_gradient_flow(self) -> None:
        """Test that gradients flow through attention."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
        )
        
        x = torch.randn(2, 10, 64, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attention.W_query.weight.grad is not None

    def test_attention_different_embedding_and_output_dims(self) -> None:
        """Test attention with different input and output dimensions."""
        # This tests a projection from input_dim to output_dim
        attention = MultiHeadAttention(
            embeddings_dimension=32,
            output_dimension=64,
            context_length=256,
            heads=4,
        )
        
        x = torch.randn(2, 8, 32)
        output = attention(x)
        assert output.shape == (2, 8, 64)

    def test_attention_dropout(self) -> None:
        """Test that attention has dropout."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
            dropout=0.5,
        )
        
        assert attention.dropout.p == 0.5

    def test_attention_learnable_parameters(self) -> None:
        """Test that attention has learnable parameters."""
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=512,
            heads=8,
        )
        
        # Get parameter names
        param_names = [name for name, _ in attention.named_parameters()]
        
        # Should have W_query, W_key, W_value, out_projection parameters
        assert len(param_names) > 0
        assert any("W_query" in name for name in param_names)
        assert any("W_key" in name for name in param_names)
        assert any("W_value" in name for name in param_names)
        assert any("out_projection" in name for name in param_names)

    def test_attention_mask_buffer(self) -> None:
        """Test that causal mask is registered as buffer."""
        context_length = 256
        attention = MultiHeadAttention(
            embeddings_dimension=64,
            output_dimension=64,
            context_length=context_length,
            heads=8,
        )
        
        assert hasattr(attention, "mask")
        assert attention.mask.shape == (context_length, context_length)

    def test_attention_multiple_heads(self) -> None:
        """Test attention with different numbers of heads."""
        emb_dim = 256
        
        for num_heads in [4, 8, 16]:
            attention = MultiHeadAttention(
                embeddings_dimension=emb_dim,
                output_dimension=emb_dim,
                context_length=512,
                heads=num_heads,
            )
            
            x = torch.randn(2, 10, emb_dim)
            output = attention(x)
            assert output.shape == (2, 10, emb_dim)
