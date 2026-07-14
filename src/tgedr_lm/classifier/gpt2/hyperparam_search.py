"""Hyperparameter search module using Optuna backend with Hugging Face Transformers.

This module provides the HyperParamSearch class for optimizing model training parameters
using Optuna integration with the Hugging Face Transformers Trainer.
"""

from collections.abc import Callable
import logging
import os
from importlib.util import find_spec

from transformers import Trainer, TrainingArguments


os.environ["TENSORBOARD_LOGGING_DIR"] = "./tensorboard_logs"
logger = logging.getLogger(__name__)


class HyperParamSearch:
    """Hyperparameter search class for optimizing model training parameters using Optuna.

    This class provides functionality to perform hyperparameter search using Optuna backend
    with Hugging Face Transformers Trainer.
    """

    def __init__(
        self, compute_metrics_func: Callable, train_args: TrainingArguments, hp_space: Callable | None = None
    ) -> None:
        """Initialize the hyperparameter search class.

        Parameters
        ----------
        compute_metrics_func : Callable
            Function to compute metrics for evaluation.
        train_args : TrainingArguments
            Hugging Face TrainingArguments for the Trainer.
        hp_space : Callable | None, optional
            Function defining the hyperparameter search space, by default None.
            If None, default hyperparameter space will be used.
        """
        self._train_args = train_args
        self._compute_metrics = compute_metrics_func
        self._hp_space = hp_space or self._get_default_hp_space()

    def _get_default_hp_space(self) -> Callable:
        """Define the default hyperparameter search space."""
        return lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "warmup_steps": trial.suggest_float("warmup_steps", 0.0, 0.2),
        }

    def search(self, model, train_dataset, val_dataset, trials: int = 8) -> dict:
        """Search for optimal hyperparameters using Optuna.

        Parameters
        ----------
        model : Any
            The model to optimize.
        train_dataset : Any
            Training dataset for hyperparameter search.
        val_dataset : Any
            Validation dataset for evaluation during search.
        trials : int, optional
            Number of trials to run, by default 8.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters found.
        """
        logger.info(f"[search|in] ({model}, {train_dataset}, {val_dataset}, {trials})")
        best_hyperparameters = {}

        if find_spec("optuna") is None:
            error_msg = "Optuna is not installed. Please install it to use hyperparameter search."
            raise ImportError(error_msg)

        logger.info("Running automatic hyperparameter search (Optuna backend)...")

        search_trainer = Trainer(
            model_init=lambda: model,
            args=self._train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

        best_run = search_trainer.hyperparameter_search(
            backend="optuna",
            direction="maximize",
            n_trials=trials,
            hp_space=self._hp_space,
            compute_objective=lambda metrics: metrics["eval_accuracy"],
        )

        best_hyperparameters = best_run.hyperparameters
        logger.info(f"[search|out] => {best_hyperparameters}")
        return best_hyperparameters
