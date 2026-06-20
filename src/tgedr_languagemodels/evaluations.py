"""Model evaluation mixins for PyTorch models.

This module provides mixin classes for evaluating PyTorch models on training and validation datasets:
- ModelEvaluatorMixin: Abstract base mixin for model evaluation
- CrossEntropyModelEvaluatorMixin: Concrete implementation using cross-entropy loss
"""

from abc import ABC, abstractmethod
import logging
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluatorMixin(ABC):
    """Mixin class for evaluating PyTorch models on training and validation datasets.

    This abstract mixin provides methods to evaluate model performance by calculating
    losses on data loaders and provides abstract methods for subclasses to implement
    specific loss calculation strategies.

    Methods
    -------
    calculate_batch_loss(input_batch, target_batch, device) -> torch.Tensor
        Abstract method to calculate loss for a single batch.
    calculate_loader_loss(data_loader, device, num_batches) -> float
        Abstract method to calculate average loss over batches in a DataLoader.
    """

    @abstractmethod
    def calculate_batch_loss(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Calculate the loss for a single batch of inputs and targets.

        Parameters
        ----------
        input_batch : torch.Tensor
            The input tensor batch.
        target_batch : torch.Tensor
            The target tensor batch.
        device : torch.device
            The device to run the model on.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """

    @abstractmethod
    def calculate_loader_accuracy(
        self, data_loader: DataLoader, device: torch.device, num_batches: int | None = None
    ) -> float:
        """Calculate the accuracy over batches in a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader containing input and target batches.
        device : torch.device
            The device to run the model on.
        num_batches : int|None, optional
            Maximum number of batches to process. If None, process all batches.

        Returns
        -------
        float
            The accuracy (fraction of correct predictions) across the batches.
        """


class CrossEntropyModelEvaluatorMixin(ModelEvaluatorMixin):
    """Mixin class for evaluating PyTorch models using cross-entropy loss.

    This mixin implements the abstract methods from ModelEvaluatorMixin to calculate
    cross-entropy loss for model evaluation on training and validation datasets.

    Methods
    -------
    calculate_batch_loss(input_batch, target_batch, device) -> torch.Tensor
        Calculate cross-entropy loss for a single batch.
    calculate_loader_loss(data_loader, device, num_batches) -> float
        Calculate average cross-entropy loss over batches in a DataLoader.
    """

    def calculate_batch_loss(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Calculate cross-entropy loss for a batch of inputs and targets.

        Parameters
        ----------
        input_batch : torch.Tensor
            The input tensor batch.
        target_batch : torch.Tensor
            The target tensor batch.
        device : torch.device
            The device to run the model on.

        Returns
        -------
        torch.Tensor
            The cross-entropy loss for the batch.
        """
        logger.debug(f"[calculate_batch_loss|in] ({input_batch}, {target_batch}, {device})")
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = self(input_batch)[:, -1, :]  # Get the logits for the last time step
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        logger.debug(f"[calculate_batch_loss|out] => {loss}")
        return loss

    def calculate_loader_accuracy(
        self, data_loader: DataLoader, device: torch.device, num_batches: int | None = None
    ) -> float:
        """Calculate the accuracy over batches in a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader containing input and target batches.
        device : torch.device
            The device to run the model on.
        num_batches : int|None, optional
            Maximum number of batches to process. If None, process all batches.

        Returns
        -------
        float
            The accuracy (fraction of correct predictions) across the batches.
        """
        logger.debug(f"[calculate_loader_accuracy|in] ({data_loader}, {device}, {num_batches})")

        was_training = self.training
        if was_training:
            self.eval()
        correct_predictions = 0
        total_predictions = 0

        if len(data_loader) == 0:
            if was_training:
                self.train()
            return float("nan")

        num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                    input_batch = input_batch.to(device)  # noqa: PLW2901
                    target_batch = target_batch.to(device)  # noqa: PLW2901
                    logits = self(input_batch)[:, -1, :]  # Get the logits for the last time step
                    predicted_labels = torch.argmax(logits, dim=-1)
                    correct_predictions += (predicted_labels == target_batch).sum().item()
                    total_predictions += target_batch.shape[0]
                else:
                    break

            result = correct_predictions / total_predictions if total_predictions > 0 else float("nan")
        if was_training:
            self.train()
        logger.debug(f"[calculate_loader_accuracy|out] => {result}")
        return result
