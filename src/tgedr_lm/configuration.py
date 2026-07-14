"""Configuration dataclasses and presets for GPT-2 language models."""

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from transformers import PretrainedConfig, TrainingArguments

from tgedr_lm.commons.errors import UnknownTrainingArgsError

GPT2_MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


class ClassifierBaseConfiguration(PretrainedConfig):
    """Hugging Face compatible configuration for GPT2Classifier.

    This wraps the BaseClassifierConfig to be compatible with PreTrainedModel.
    """

    def __init__(
        self,
        vocabulary_size: int = 50257,
        embeddings_dimension: int = 768,
        context_length: int = 1024,
        n_layers: int = 12,
        drop_rate: float = 0.1,
        stride: int = 1,
        n_heads: int = 12,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        n_classes: int = 2,
        model_type: str = "classifier-gpt2",
        **kwargs: any,
    ) -> None:
        """Initialize GPT2 classifier configuration.

        Parameters
        ----------
        vocabulary_size : int
            Number of tokens in the vocabulary.
        embeddings_dimension : int
            Dimension of token and positional embeddings.
        context_length : int
            Maximum number of tokens in a sequence.
        n_layers : int
            Number of transformer blocks.
        drop_rate : float
            Dropout probability for regularization.
        stride : int
            Stride used for sliding window tokenization.
        n_heads : int
            Number of attention heads per transformer block.
        qkv_bias : bool
            Whether to use bias in query, key, and value projections.
        n_classes : int
            Number of output classes for classification.
        model_type : str
            Model type identifier for Hugging Face compatibility.
        **kwargs : any
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embeddings_dimension = embeddings_dimension
        self.context_length = context_length
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.stride = stride
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.n_classes = n_classes
        self.model_type = model_type

    def to_dict(self) -> dict[str, Any]:
        """Return all dataclass member names and values as a dictionary."""
        return asdict(self)

    def update_from_dict(self, updates: dict[str, Any]) -> None:
        """Update dataclass members from a dictionary of member names to values."""
        valid_fields = {field.name for field in fields(self)}
        unknown_fields = set(updates) - valid_fields
        if unknown_fields:
            raise UnknownTrainingArgsError(unknown_fields)

        for key, value in updates.items():
            setattr(self, key, value)

    def set(self, key, value) -> None:  # noqa: D102
        setattr(self, key, value)


@dataclass
class TrainingArgs:
    """Training arguments for the GPT-2 classifier.

    Attributes
    ----------
    eval_strategy : str
        Strategy for evaluation (e.g., 'epoch', 'steps').
    fp16 : bool
        Whether to use mixed precision training (FP16).
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients before updating model parameters.
    greater_is_better : bool
        Whether higher metric values indicate better performance.
    hub_model_id : str
        Model ID for pushing to Hugging Face Hub.
    hub_strategy : str
        Strategy for pushing to Hugging Face Hub (e.g., 'end', 'every_save').
    learning_rate : float
        Initial learning rate for optimizer.
    load_best_model_at_end : bool
        Whether to load the best model at the end of training based on evaluation metrics.
    logging_steps : int
        Frequency of logging in steps.
    logging_strategy : str
        Strategy for logging (e.g., 'steps', 'epoch').
    metric_for_best_model : str
        Metric used to determine the best model during training.
    num_train_epochs : int
        Number of training epochs.
    output_dir : str
        Directory for saving model checkpoints and outputs.
    per_device_eval_batch_size : int
        Batch size per device during evaluation.
    per_device_train_batch_size : int
        Batch size per device during training.
    remove_unused_columns : bool
        Whether to remove unused columns from the dataset.
    report_to : str
        Reporting tool for logging (e.g., 'tensorboard', 'wandb', 'none').
    save_only_model : bool
        Whether to save only the model or the entire training state.
    save_strategy : str
        Strategy for saving checkpoints (e.g., 'epoch', 'steps').
    save_total_limit : int
        Maximum number of checkpoints to keep.
    seed : int
        Random seed for reproducibility.
    warmup_steps : float
        Number of warmup steps for learning rate scheduler.
    weight_decay : float
        Weight decay for optimizer regularization.
    hub_model_id : str
        Model ID for pushing to Hugging Face Hub.
    hub_strategy : str
        Strategy for pushing to Hugging Face Hub (e.g., 'end', 'every_save').
    """

    eval_strategy: str = "epoch"
    fp16: bool = torch.cuda.is_available()
    gradient_accumulation_steps: int = 1
    greater_is_better: bool = True
    hub_model_id: str = ""
    hub_strategy: str = "end"
    learning_rate: float = 2e-4
    load_best_model_at_end: bool = True
    logging_steps: int = 5
    logging_strategy: str = "steps"
    metric_for_best_model: str = "accuracy"
    num_train_epochs: int = 5
    output_dir: str = "./experiment_results"
    per_device_eval_batch_size: int = 16
    per_device_train_batch_size: int = 8
    remove_unused_columns: bool = False
    report_to: str = "none"
    save_only_model: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    seed: int = 53
    warmup_steps: float = 0.1
    weight_decay: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        """Return all dataclass member names and values as a dictionary."""
        return asdict(self)

    def update_from_dict(self, updates: dict[str, Any]) -> None:
        """Update dataclass members from a dictionary of member names to values."""
        valid_fields = {field.name for field in fields(self)}
        unknown_fields = set(updates) - valid_fields
        if unknown_fields:
            raise UnknownTrainingArgsError(unknown_fields)

        for key, value in updates.items():
            setattr(self, key, value)

    def set(self, key, value) -> None:  # noqa: D102
        setattr(self, key, value)

    def to_training_arguments(self) -> TrainingArguments:
        """Package all configured values as a transformers TrainingArguments object."""
        return TrainingArguments(**self.to_dict())
