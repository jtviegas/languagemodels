"""Example: Using GPT2Classifier with Hugging Face Trainer.

This example demonstrates how to train GPT2Classifier using the Hugging Face
Trainer API, which provides features like distributed training, gradient
accumulation, mixed precision, and comprehensive logging.
"""

import torch
import tiktoken
import numpy as np
from importlib.util import find_spec

from tgedr_languagemodels.classifier.gpt2.configuration import ClassifierConfiguration
from tgedr_languagemodels.classifier.gpt2.text_dataset import TextDataset
from transformers import Trainer, TrainingArguments
from tgedr_languagemodels.classifier.gpt2.model import GPT2Classifier





def example_trainer_usage():
    """Example: Train GPT2Classifier using Hugging Face Trainer.

    This example shows the basic setup. For production use:
    1. Use your actual datasets (TextDataset, ClassifierDataLoader)
    2. Configure TrainingArguments for your hardware
    3. Add custom callbacks for logging/monitoring
    4. Use a proper data collator if needed
    """

    # 1. Configure the model
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

    def model_init():
        """Create a fresh model instance (required for hyperparameter search)."""
        return GPT2Classifier(config)

    # 3. Prepare datasets (example with dummy data)
    # In reality, use your TextDataset/ClassifierDataLoader
    train_texts = ["sample text 1", "sample text 2", "sample text 3"] * 100
    train_labels = [0, 1, 2] * 100

    val_texts = ["validation text 1", "validation text 2", "validation text 3"] * 20
    val_labels = [0, 1, 2] * 20

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = TextDataset(tokenizer=tokenizer, texts=train_texts, labels=train_labels)
    val_dataset = TextDataset(tokenizer=tokenizer, texts=val_texts, labels=val_labels)

    def compute_metrics(eval_pred):
        """Compute accuracy using classifier logits from the last token."""
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits[:, -1, :], axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    # 4. Optionally search for better training arguments
    best_hyperparameters = {}
    try:
        if find_spec("optuna") is None:
            raise ImportError

        print("Running automatic hyperparameter search (Optuna backend)...")

        search_args = TrainingArguments(
            output_dir="./experiment_results_hp_search",
            logging_dir="./logs",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=3e-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to="none",
            seed=42,
        )

        search_trainer = Trainer(
            model_init=model_init,
            args=search_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        best_run = search_trainer.hyperparameter_search(
            backend="optuna",
            direction="maximize",
            n_trials=8,
            hp_space=lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            },
            compute_objective=lambda metrics: metrics["eval_accuracy"],
        )

        best_hyperparameters = best_run.hyperparameters
        print(f"Best hyperparameters found: {best_hyperparameters}")

    except ImportError:
        print("Optuna is not installed. Using default TrainingArguments.")

    training_kwargs = {
        "output_dir": "./experiment_results",
        "logging_dir": "./logs",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
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
        "seed": 42,
        "remove_unused_columns": False,
    }

    for key in [
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "warmup_ratio",
    ]:
        if key in best_hyperparameters:
            training_kwargs[key] = best_hyperparameters[key]

    training_kwargs["num_train_epochs"] = int(training_kwargs["num_train_epochs"])
    training_kwargs["per_device_train_batch_size"] = int(training_kwargs["per_device_train_batch_size"])

    training_args = TrainingArguments(**training_kwargs)

    # 5. Create Trainer
    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # Optional: add callbacks for custom behavior
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    final_metrics = trainer.evaluate()
    print(f"Final eval accuracy: {final_metrics.get('eval_accuracy')}")
    print(f"Final eval loss: {final_metrics.get('eval_loss')}")

    # 7. Save model
    # model.save_pretrained("./final_model")

    print("Training complete!")

    return trainer



if __name__ == "__main__":
    # Note: This is an example structure. Uncomment to run:
    # example_trainer_usage()
    print("See the example code in this file for usage patterns.")
    print("\nBasic usage:")
    print("  from tgedr_languagemodels import create_trainer_ready_classifier")
    print("  from transformers import Trainer, TrainingArguments")
    print("  ")
    print("  model = create_trainer_ready_classifier(config)")
    print("  trainer = Trainer(model=model, args=args, train_dataset=ds, ...)")
    print("  trainer.train()")
    example_trainer_usage()
