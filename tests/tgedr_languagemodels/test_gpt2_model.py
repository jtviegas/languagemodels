"""Unit tests for the GPT-2 model module."""

import pytest
import torch
from tgedr_languagemodels.gpt2.model import GPT2Model
from tgedr_languagemodels.configuration import BaseModelConfig


class TestGPT2Model:
    """Test suite for GPT2Model."""

    def get_test_config(self) -> BaseModelConfig:
        """Get a test configuration."""
        return BaseModelConfig(
            vocabulary_size=1000,
            embeddings_dimension=128,
            context_length=64,
            n_layers=2,
            drop_rate=0.1,
            stride=1,
            n_heads=8,
        )

    def test_model_initialization(self) -> None:
        """Test GPT2Model initialization."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        assert model.tok_emb is not None
        assert model.pos_emb is not None
        assert model.drop_emb is not None
        assert model.trf_blocks is not None
        assert model.final_norm is not None
        assert model.out_head is not None

    def test_model_embedding_dimensions(self) -> None:
        """Test that embeddings have correct dimensions."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        assert model.tok_emb.num_embeddings == cfg.vocabulary_size
        assert model.tok_emb.embedding_dim == cfg.embeddings_dimension
        assert model.pos_emb.num_embeddings == cfg.context_length
        assert model.pos_emb.embedding_dim == cfg.embeddings_dimension

    def test_model_forward_shape(self) -> None:
        """Test GPT2Model forward pass shape."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        batch_size, seq_len = 2, 10
        idx = torch.randint(0, cfg.vocabulary_size, (batch_size, seq_len))
        output = model(idx)
        
        expected_shape = (batch_size, seq_len, cfg.vocabulary_size)
        assert output.shape == expected_shape

    def test_model_output_logits(self) -> None:
        """Test that model outputs logits."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        idx = torch.randint(0, cfg.vocabulary_size, (1, 5))
        logits = model(idx)
        
        # Logits should contain real values
        assert torch.isfinite(logits).all()

    def test_model_different_batch_sizes(self) -> None:
        """Test model with different batch sizes."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        for batch_size in [1, 2, 4]:
            idx = torch.randint(0, cfg.vocabulary_size, (batch_size, 10))
            output = model(idx)
            assert output.shape == (batch_size, 10, cfg.vocabulary_size)

    def test_model_different_sequence_lengths(self) -> None:
        """Test model with different sequence lengths."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        for seq_len in [1, 5, 10, 20]:
            idx = torch.randint(0, cfg.vocabulary_size, (2, seq_len))
            output = model(idx)
            assert output.shape == (2, seq_len, cfg.vocabulary_size)

    def test_model_number_of_layers(self) -> None:
        """Test that model has correct number of layers."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        assert len(model.trf_blocks) == cfg.n_layers

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        idx = torch.randint(0, cfg.vocabulary_size, (2, 5), requires_grad=False)
        idx = idx.long()  # Embeddings require long tensors
        output = model(idx)
        loss = output.sum()
        loss.backward()
        
        # Check that embeddings have gradients
        assert model.tok_emb.weight.grad is not None

    def test_model_dropout_rate(self) -> None:
        """Test that model uses correct dropout rate."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        assert model.drop_emb.p == cfg.drop_rate

    def test_model_eval_mode(self) -> None:
        """Test model in eval mode."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        model.eval()
        
        idx = torch.randint(0, cfg.vocabulary_size, (2, 5))
        output = model(idx)
        assert output.shape == (2, 5, cfg.vocabulary_size)

    def test_model_train_mode(self) -> None:
        """Test model in train mode."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        model.train()
        
        idx = torch.randint(0, cfg.vocabulary_size, (2, 5))
        output = model(idx)
        assert output.shape == (2, 5, cfg.vocabulary_size)

    def test_model_output_head_dimension(self) -> None:
        """Test that output head has correct output dimension."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        assert model.out_head.out_features == cfg.vocabulary_size
        assert model.out_head.in_features == cfg.embeddings_dimension

    def test_model_parameters(self) -> None:
        """Test that model has learnable parameters."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        params = list(model.parameters())
        assert len(params) > 0

    def test_model_device_handling(self) -> None:
        """Test model on CPU."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        model = model.cpu()
        
        idx = torch.randint(0, cfg.vocabulary_size, (1, 5))
        output = model(idx)
        assert output.device.type == 'cpu'

    def test_model_with_different_vocab_sizes(self) -> None:
        """Test model with different vocabulary sizes."""
        for vocab_size in [256, 1000, 10000]:
            cfg = BaseModelConfig(
                vocabulary_size=vocab_size,
                embeddings_dimension=128,
                context_length=64,
                n_layers=1,
                drop_rate=0.1,
                stride=1,
                n_heads=8,
            )
            model = GPT2Model(cfg)
            
            idx = torch.randint(0, vocab_size, (1, 5))
            output = model(idx)
            assert output.shape[2] == vocab_size

    def test_model_deterministic_with_seed(self) -> None:
        """Test that model produces same outputs with same seed."""
        cfg = self.get_test_config()
        
        torch.manual_seed(42)
        model1 = GPT2Model(cfg)
        model1.eval()
        
        torch.manual_seed(42)
        model2 = GPT2Model(cfg)
        model2.eval()
        
        idx = torch.randint(0, cfg.vocabulary_size, (1, 5))
        
        with torch.no_grad():
            output1 = model1(idx)
            output2 = model2(idx)
        
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)

    def test_model_no_bias_in_output_head(self) -> None:
        """Test that output head has no bias."""
        cfg = self.get_test_config()
        model = GPT2Model(cfg)
        
        # Model is constructed with bias=False for output layer
        assert model.out_head.bias is None
