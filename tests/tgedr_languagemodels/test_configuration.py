"""Unit tests for the configuration module."""

import pytest
from tgedr_languagemodels.configuration import BaseModelConfig, GPT2_MODEL_CONFIGS


class TestBaseModelConfig:
    """Test suite for BaseModelConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating a BaseModelConfig instance."""
        config = BaseModelConfig(
            vocabulary_size=50257,
            embeddings_dimension=768,
            context_length=1024,
            n_layers=12,
            drop_rate=0.1,
            stride=1,
            n_heads=12,
        )
        assert config.vocabulary_size == 50257
        assert config.embeddings_dimension == 768
        assert config.context_length == 1024
        assert config.n_layers == 12
        assert config.drop_rate == 0.1
        assert config.stride == 1
        assert config.n_heads == 12
        assert config.qkv_bias is False

    def test_config_with_qkv_bias(self) -> None:
        """Test creating a BaseModelConfig with qkv_bias."""
        config = BaseModelConfig(
            vocabulary_size=50257,
            embeddings_dimension=768,
            context_length=1024,
            n_layers=12,
            drop_rate=0.1,
            stride=1,
            n_heads=12,
            qkv_bias=True,
        )
        assert config.qkv_bias is True

    def test_config_default_qkv_bias(self) -> None:
        """Test that qkv_bias defaults to False."""
        config = BaseModelConfig(
            vocabulary_size=50257,
            embeddings_dimension=768,
            context_length=1024,
            n_layers=12,
            drop_rate=0.1,
            stride=1,
            n_heads=12,
        )
        assert config.qkv_bias is False

    def test_config_is_dataclass(self) -> None:
        """Test that BaseModelConfig is a dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(BaseModelConfig)

    def test_config_repr(self) -> None:
        """Test string representation of config."""
        config = BaseModelConfig(
            vocabulary_size=50257,
            embeddings_dimension=768,
            context_length=1024,
            n_layers=12,
            drop_rate=0.1,
            stride=1,
            n_heads=12,
        )
        repr_str = repr(config)
        assert "BaseModelConfig" in repr_str
        assert "50257" in repr_str


class TestGPT2ModelConfigs:
    """Test suite for GPT2_MODEL_CONFIGS constant."""

    def test_gpt2_configs_exist(self) -> None:
        """Test that GPT2_MODEL_CONFIGS is defined."""
        assert isinstance(GPT2_MODEL_CONFIGS, dict)
        assert len(GPT2_MODEL_CONFIGS) > 0

    def test_gpt2_small_config(self) -> None:
        """Test GPT-2 small model configuration."""
        assert "gpt2-small (124M)" in GPT2_MODEL_CONFIGS
        small_config = GPT2_MODEL_CONFIGS["gpt2-small (124M)"]
        assert small_config["emb_dim"] == 768
        assert small_config["n_layers"] == 12
        assert small_config["n_heads"] == 12

    def test_gpt2_medium_config(self) -> None:
        """Test GPT-2 medium model configuration."""
        assert "gpt2-medium (355M)" in GPT2_MODEL_CONFIGS
        medium_config = GPT2_MODEL_CONFIGS["gpt2-medium (355M)"]
        assert medium_config["emb_dim"] == 1024
        assert medium_config["n_layers"] == 24
        assert medium_config["n_heads"] == 16

    def test_gpt2_large_config(self) -> None:
        """Test GPT-2 large model configuration."""
        assert "gpt2-large (774M)" in GPT2_MODEL_CONFIGS
        large_config = GPT2_MODEL_CONFIGS["gpt2-large (774M)"]
        assert large_config["emb_dim"] == 1280
        assert large_config["n_layers"] == 36
        assert large_config["n_heads"] == 20

    def test_gpt2_xl_config(self) -> None:
        """Test GPT-2 XL model configuration."""
        assert "gpt2-xl (1558M)" in GPT2_MODEL_CONFIGS
        xl_config = GPT2_MODEL_CONFIGS["gpt2-xl (1558M)"]
        assert xl_config["emb_dim"] == 1600
        assert xl_config["n_layers"] == 48
        assert xl_config["n_heads"] == 25

    def test_all_configs_have_required_keys(self) -> None:
        """Test that all configs have the required keys."""
        required_keys = {"emb_dim", "n_layers", "n_heads"}
        for config_name, config in GPT2_MODEL_CONFIGS.items():
            assert required_keys.issubset(config.keys()), f"Missing keys in {config_name}"

    def test_config_values_are_positive(self) -> None:
        """Test that all config values are positive integers."""
        for config_name, config in GPT2_MODEL_CONFIGS.items():
            for key, value in config.items():
                assert isinstance(value, int), f"{key} in {config_name} is not an int"
                assert value > 0, f"{key} in {config_name} is not positive"
