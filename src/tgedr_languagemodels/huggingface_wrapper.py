"""Hugging Face Trainer-compatible wrapper for GPT2Classifier."""

import torch
from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import asdict

from tgedr_languagemodels.configuration import BaseClassifierConfig
from tgedr_languagemodels.gpt2.classifier import GPT2Classifier


class GPT2ClassifierConfig(PretrainedConfig):
    """Hugging Face compatible configuration for GPT2Classifier.

    This wraps the BaseClassifierConfig to be compatible with PreTrainedModel.
    """

    model_type = "gpt2-classifier"

    def __init__(
        self,
        vocabulary_size: int = 50257,
        embeddings_dimension: int = 768,
        context_length: int = 1024,
        n_layers: int = 12,
        drop_rate: float = 0.1,
        stride: int = 1,
        n_heads: int = 12,
        qkv_bias: bool = False,
        n_classes: int = 2,
        **kwargs,
    ):
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


class GPT2ClassifierForHuggingFace(PreTrainedModel):
    """GPT2Classifier wrapped for Hugging Face Trainer compatibility.

    This wrapper makes GPT2Classifier compatible with Hugging Face's Trainer API
    by converting between the Trainer's expected interface and the model's native format.

    Parameters
    ----------
    config : GPT2ClassifierConfig
        Model configuration.
    """

    config_class = GPT2ClassifierConfig

    def __init__(self, config: GPT2ClassifierConfig) -> None:
        """Initialize the wrapped classifier.

        Parameters
        ----------
        config : GPT2ClassifierConfig
            Model configuration.
        """
        super().__init__(config)

        # Convert Hugging Face config to our BaseClassifierConfig
        base_config = BaseClassifierConfig(
            vocabulary_size=config.vocabulary_size,
            embeddings_dimension=config.embeddings_dimension,
            context_length=config.context_length,
            n_layers=config.n_layers,
            drop_rate=config.drop_rate,
            stride=config.stride,
            n_heads=config.n_heads,
            qkv_bias=config.qkv_bias,
            n_classes=config.n_classes,
        )

        # Initialize the underlying GPT2Classifier
        self.classifier = GPT2Classifier(base_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Forward pass compatible with Hugging Face Trainer.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token indices of shape (batch_size, sequence_length).
        attention_mask : torch.Tensor, optional
            Attention mask (not currently used by GPT2Classifier).
        labels : torch.Tensor, optional
            Target class labels for computing loss.
        **kwargs
            Additional arguments (ignored for compatibility).

        Returns
        -------
        dict
            Dictionary containing 'logits' and optionally 'loss'.
        """
        # Move inputs to the model's device
        device = self.device
        input_ids = input_ids.to(device)

        # Get logits from classifier
        logits = self.classifier(input_ids)

        # Initialize output dict
        outputs = {"logits": logits}

        # Compute loss if labels are provided
        if labels is not None:
            labels = labels.to(device)
            loss = self.classifier.calculate_batch_loss(input_ids, labels, device)
            outputs["loss"] = loss

        return outputs


def create_trainer_ready_classifier(config: BaseClassifierConfig) -> GPT2ClassifierForHuggingFace:
    """Create a Trainer-ready GPT2Classifier wrapper.

    Parameters
    ----------
    config : BaseClassifierConfig
        Base classifier configuration.

    Returns
    -------
    GPT2ClassifierForHuggingFace
        Wrapped classifier ready for Hugging Face Trainer.

    Example
    -------
    >>> from tgedr_languagemodels.configuration import BaseClassifierConfig
    >>> from tgedr_languagemodels.huggingface_wrapper import create_trainer_ready_classifier
    >>> from transformers import Trainer, TrainingArguments
    >>>
    >>> config = BaseClassifierConfig(
    ...     vocabulary_size=50257,
    ...     embeddings_dimension=768,
    ...     context_length=1024,
    ...     n_layers=12,
    ...     drop_rate=0.1,
    ...     stride=1,
    ...     n_heads=12,
    ...     n_classes=2,
    ... )
    >>> model = create_trainer_ready_classifier(config)
    >>>
    >>> training_args = TrainingArguments(
    ...     output_dir="./results",
    ...     num_train_epochs=3,
    ...     per_device_train_batch_size=16,
    ...     per_device_eval_batch_size=16,
    ...     save_steps=100,
    ...     eval_steps=100,
    ... )
    >>> trainer = Trainer(
    ...     model=model,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=eval_dataset,
    ... )
    >>> trainer.train()
    """
    # Convert BaseClassifierConfig to GPT2ClassifierConfig
    hf_config = GPT2ClassifierConfig(
        vocabulary_size=config.vocabulary_size,
        embeddings_dimension=config.embeddings_dimension,
        context_length=config.context_length,
        n_layers=config.n_layers,
        drop_rate=config.drop_rate,
        stride=config.stride,
        n_heads=config.n_heads,
        qkv_bias=config.qkv_bias,
        n_classes=config.n_classes,
    )

    return GPT2ClassifierForHuggingFace(hf_config)
