import logging
import os
import torch
import numpy as np
from importlib.util import find_spec

from transformers import Trainer, TrainingArguments


os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
logger = logging.getLogger(__name__)

class HyperParamSearch:
  
  def __init__(self, compute_metrics_func, train_args: TrainingArguments = None):
    """Initialize the hyperparameter search class.

    Parameters
    ----------
    train_args : TrainingArguments, optional
        Hugging Face TrainingArguments for the Trainer, by default None.
        If None, default arguments will be used.
    """
    self.train_args = train_args or self._get_default_arguments()
    self._compute_metrics = compute_metrics_func

  
  def _get_default_arguments(self) -> TrainingArguments:
    """Get default training arguments for hyperparameter search."""
    return TrainingArguments(
        output_dir=".experiment_results_hp_search",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=3e-4,
        warmup_steps=0.05,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        seed=53,
    )


  def search(self, model, train_dataset, val_dataset, trials: int = 8):
    logger.info(f"[search|in] ({model}, {train_dataset}, {val_dataset}, {trials})")
    best_hyperparameters = {}
    
    if find_spec("optuna") is None:
        raise ImportError("Optuna is not installed. Please install it to use hyperparameter search.")

    logger.info("Running automatic hyperparameter search (Optuna backend)...")

    search_trainer = Trainer(
        model_init=lambda: model,
        args=self.train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=self._compute_metrics,
    )

    best_run = search_trainer.hyperparameter_search(
        backend="optuna",
        direction="maximize",
        n_trials=trials,
        hp_space=lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "warmup_steps": trial.suggest_float("warmup_steps", 0.0, 0.2),
        },
        compute_objective=lambda metrics: metrics["eval_accuracy"],
    )

    best_hyperparameters = best_run.hyperparameters
    logger.info(f"[search|out] => {best_hyperparameters}")
    return best_hyperparameters


  
