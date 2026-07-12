import logging

import pytest
from tgedr_lm.classifier.gpt2.model import GPT2Classifier
from tgedr_lm.configuration import ClassifierBaseConfiguration, TrainingArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from pytest import fixture
import torch
import tiktoken
from importlib.util import find_spec

from tgedr_lm.classifier.gpt2.hyperparam_search import HyperParamSearch
from tgedr_lm.classifier.text_dataset import TextDataset
from transformers import Trainer, TrainingArguments

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


def test_train(data):

    train_dataset, val_dataset = data
    training_args = TrainingArgs()

    hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=training_args.to_training_arguments())
    hyperparameters = hp_search.search(model=GPT2Classifier(ClassifierBaseConfiguration(n_classes=3)), 
                                       train_dataset=train_dataset, val_dataset=val_dataset, 
                                       trials=3)
    
    
    for key in [
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "warmup_steps",
    ]:
        if key in hyperparameters:
            training_args.set(key, hyperparameters[key])
    # training_args.set("hub_model_id", "jtviegas/gpt2classifier")

    model = GPT2Classifier(ClassifierBaseConfiguration(n_classes=3))

    # training_kwargs["num_train_epochs"] = int(training_kwargs["num_train_epochs"])
    # training_kwargs["per_device_train_batch_size"] = int(training_kwargs["per_device_train_batch_size"])

    trainer = Trainer(
        model=model,
        args=training_args.to_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=model.compute_metrics,
        # Optional: add callbacks for custom behavior
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    accuracy = final_metrics.get("eval_accuracy")
    loss = final_metrics.get("eval_loss")
    assert accuracy is not None and accuracy > 0.9
    assert loss is not None and loss < 0.3

    # 7. Save model
    # model.save_pretrained("./final_model")

    # print("Training complete!")
