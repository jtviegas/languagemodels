import logging

from tgedr_lm.classifier.gpt2.model import GPT2Classifier
from tgedr_lm.configuration import ClassifierBaseConfiguration, TrainingArgs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

import pytest
from pytest import fixture
import tiktoken
from importlib.util import find_spec

from tgedr_lm.classifier.gpt2.hyperparam_search import HyperParamSearch
from tgedr_lm.classifier.text_dataset import TextDataset


@fixture
def model():
    """Create a fresh model instance (required for hyperparameter search)."""
    config = ClassifierBaseConfiguration(
        vocabulary_size=50257,
        embeddings_dimension=768,
        context_length=1024,
        n_layers=12,
        drop_rate=0.1,
        stride=1,
        n_heads=12,
        n_classes=3,
    )
    return GPT2Classifier(config)

@fixture
def data():
    """Prepare datasets (example with dummy data)."""
    train_texts = ["sample text 1", "sample text 2", "sample text 3"] * 30
    train_labels = [0, 1, 2] * 30

    val_texts = ["validation text 1", "validation text 2", "validation text 3"] * 10
    val_labels = [0, 1, 2] * 10

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = TextDataset(tokenizer=tokenizer, texts=train_texts, labels=train_labels)
    val_dataset = TextDataset(tokenizer=tokenizer, texts=val_texts, labels=val_labels)

    return train_dataset, val_dataset

def test_hyperparameters_default(model, data):
    train_dataset, val_dataset = data
    training_args = TrainingArgs()
    hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=training_args.to_training_arguments())
    result = hp_search.search(model=model, train_dataset=train_dataset, val_dataset=val_dataset, trials=3)
    assert result is not None
    assert result["num_train_epochs"] >= 2
    assert result["per_device_train_batch_size"] > 2
    assert result["learning_rate"] < 1e-3
    assert result["weight_decay"] < 0.1
    assert result["warmup_steps"] > 0.01


def test_search_raises_if_optuna_is_missing(monkeypatch):
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.hyperparam_search.find_spec", lambda _: None)

    training_args = TrainingArgs()
    hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=training_args.to_training_arguments())

    with pytest.raises(ImportError, match="Optuna is not installed"):
        hp_search.search(model=None, train_dataset=None, val_dataset=None, trials=1)

def test_hyperparameters_with_hp_space(model, data):
    train_dataset, val_dataset = data
    
    hp_space = lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 12),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 24]),
            "warmup_steps": trial.suggest_float("warmup_steps", 0.0, 0.4),
        }
    training_args = TrainingArgs()
    hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=training_args.to_training_arguments(), hp_space=hp_space)
    result = hp_search.search(model=model, train_dataset=train_dataset, val_dataset=val_dataset, trials=3)
    assert result is not None
    assert result["num_train_epochs"] >= 2
    assert result["per_device_train_batch_size"] > 2
    assert result["learning_rate"] < 1e-3
    assert result["weight_decay"] < 0.1
    assert result["warmup_steps"] > 0.01
