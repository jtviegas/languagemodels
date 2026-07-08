import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from pytest import fixture
import torch
import tiktoken
from importlib.util import find_spec

from tgedr_languagemodels.classifier.gpt2.configuration import ClassifierConfiguration
from tgedr_languagemodels.classifier.gpt2.hyperparam_search import HyperParamSearch
from tgedr_languagemodels.classifier.gpt2.text_dataset import TextDataset
from transformers import Trainer, TrainingArguments
from tgedr_languagemodels.classifier.gpt2.model import GPT2Classifier


@fixture
def model():
    """Create a fresh model instance (required for hyperparameter search)."""
    config = ClassifierConfiguration(
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
    train_texts = ["sample text 1", "sample text 2", "sample text 3"] * 100
    train_labels = [0, 1, 2] * 100

    val_texts = ["validation text 1", "validation text 2", "validation text 3"] * 20
    val_labels = [0, 1, 2] * 20

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = TextDataset(tokenizer=tokenizer, texts=train_texts, labels=train_labels)
    val_dataset = TextDataset(tokenizer=tokenizer, texts=val_texts, labels=val_labels)

    return train_dataset, val_dataset

def search_hyperparameters(model, data):
    train_dataset, val_dataset = data
    hp_search = HyperParamSearch(GPT2Classifier.compute_metrics)
    return hp_search.search(model=model, train_dataset=train_dataset, val_dataset=val_dataset, trials=3)



def test_hyperparameter_search(model, data):
    best_hyperparameters = search_hyperparameters(model, data)
    assert best_hyperparameters is not None
    assert best_hyperparameters["num_train_epochs"] > 1
    assert best_hyperparameters["per_device_train_batch_size"] > 1
    assert (best_hyperparameters["learning_rate"] - 7.114212060327504e-05) < 1e-1
    assert (best_hyperparameters["weight_decay"] - 0.013980743507010574) < 1e-1
    assert (best_hyperparameters["warmup_steps"] - 0.014940532738428481) < 1e-1


def test_train(model, data):

    best_hyperparameters = search_hyperparameters(model, data)
    train_dataset, val_dataset = data

    training_kwargs = {
        "output_dir": "./experiment_results",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 0.1,
        "gradient_accumulation_steps": 1,
        "fp16": torch.cuda.is_available(),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_strategy": "steps",
        "logging_steps": 5,
        "report_to": "none",
        "seed": 53,
        "remove_unused_columns": False,
    }

    for key in [
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "warmup_steps",
    ]:
        if key in best_hyperparameters:
            training_kwargs[key] = best_hyperparameters[key]

    training_kwargs["num_train_epochs"] = int(training_kwargs["num_train_epochs"])
    training_kwargs["per_device_train_batch_size"] = int(training_kwargs["per_device_train_batch_size"])

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=GPT2Classifier.compute_metrics,
        # Optional: add callbacks for custom behavior
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    assert final_metrics.get("eval_accuracy") is not None
    assert final_metrics.get("eval_loss") is not None
    assert final_metrics.get("eval_accuracy") == 1.0
    assert final_metrics.get("eval_loss") < 1e-6


    # 7. Save model
    # model.save_pretrained("./final_model")

    # print("Training complete!")
