"""Unit tests for ClassifierBaseConfiguration and TrainingArgs."""

import pytest
from transformers import TrainingArguments

from tgedr_lm.commons.errors import UnknownTrainingArgsError
from tgedr_lm.configuration import ClassifierBaseConfiguration, TrainingArgs
from tgedr_lm.configuration import GPT2_MODEL_CONFIGS


def test_to_training_arguments_packages_all_fields() -> None:
    args = TrainingArgs(
        output_dir="./tmp",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
    )

    training_arguments = args.to_training_arguments()

    assert isinstance(training_arguments, TrainingArguments)
    assert training_arguments.output_dir.endswith("tmp")
    assert training_arguments.num_train_epochs == 3
    assert training_arguments.per_device_train_batch_size == 4
    assert training_arguments.learning_rate == 1e-4


def test_build_training_arguments_applies_updates() -> None:
    args = TrainingArgs()
    args.update_from_dict(
        {
            "output_dir": "./custom",
            "learning_rate": 2e-4,
            "num_train_epochs": 7,
        }
    )
    training_arguments = args.to_training_arguments()

    assert isinstance(training_arguments, TrainingArguments)
    assert training_arguments.output_dir.endswith("custom")
    assert training_arguments.learning_rate == 2e-4
    assert training_arguments.num_train_epochs == 7


def test_build_training_arguments_raises_for_unknown_update_key() -> None:
    try:
        TrainingArgs().update_from_dict({"unknown_key": 1})
    except UnknownTrainingArgsError:
        assert True
    else:
        assert False


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


class TestTrainingArgs:
    """Test TrainingArgs behavior."""

    def test_to_training_arguments_returns_transformers_instance(self) -> None:
        args = TrainingArgs(output_dir="./tmp", learning_rate=1e-4, num_train_epochs=3)

        training_arguments = args.to_training_arguments()

        assert isinstance(training_arguments, TrainingArguments)
        assert training_arguments.output_dir.endswith("tmp")
        assert training_arguments.learning_rate == 1e-4
        assert training_arguments.num_train_epochs == 3

    def test_update_from_dict_and_set_apply_changes(self) -> None:
        args = TrainingArgs()

        args.update_from_dict({"learning_rate": 2e-4, "num_train_epochs": 7})
        args.set("per_device_train_batch_size", 16)

        assert args.learning_rate == 2e-4
        assert args.num_train_epochs == 7
        assert args.per_device_train_batch_size == 16

    def test_update_from_dict_raises_for_unknown_key(self) -> None:
        args = TrainingArgs()

        with pytest.raises(UnknownTrainingArgsError):
            args.update_from_dict({"unknown_key": 1})


class TestClassifierBaseConfiguration:
    """Test ClassifierBaseConfiguration behavior."""

    def test_initialization_sets_classifier_fields(self) -> None:
        cfg = ClassifierBaseConfiguration(
            vocabulary_size=100,
            embeddings_dimension=32,
            context_length=16,
            n_layers=2,
            drop_rate=0.2,
            stride=2,
            n_heads=4,
            qkv_bias=True,
            n_classes=3,
            model_type="classifier-gpt2",
        )

        assert cfg.vocabulary_size == 100
        assert cfg.embeddings_dimension == 32
        assert cfg.context_length == 16
        assert cfg.n_layers == 2
        assert cfg.drop_rate == 0.2
        assert cfg.stride == 2
        assert cfg.n_heads == 4
        assert cfg.qkv_bias is True
        assert cfg.n_classes == 3
        assert cfg.model_type == "classifier-gpt2"

    def test_set_updates_known_field(self) -> None:
        cfg = ClassifierBaseConfiguration(n_classes=2)

        cfg.set("n_classes", 5)

        assert cfg.n_classes == 5

    def test_to_dict_returns_pretrained_config_dictionary(self) -> None:
        cfg = ClassifierBaseConfiguration()

        cfg_dict = cfg.to_dict()
        assert isinstance(cfg_dict, dict)
        assert "transformers_version" in cfg_dict

    def test_update_from_dict_raises_for_unknown_key(self) -> None:
        cfg = ClassifierBaseConfiguration()

        with pytest.raises(UnknownTrainingArgsError):
            cfg.update_from_dict({"n_classes": 4})

    def test_update_from_dict_updates_known_pretrained_config_field(self) -> None:
        cfg = ClassifierBaseConfiguration()

        cfg.update_from_dict({"transformers_version": "test-version"})

        assert cfg.transformers_version == "test-version"
