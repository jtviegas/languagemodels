"""Example: Using GPT2Classifier with Hugging Face Trainer.

This example demonstrates how to train GPT2Classifier using the Hugging Face
Trainer API, which provides features like distributed training, gradient
accumulation, mixed precision, and comprehensive logging.
"""

import torch
import tiktoken
import numpy as np

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
        n_classes=2,
    )

    # 2. Create Trainer-ready model
    model = GPT2Classifier(config)

    # 3. Prepare datasets (example with dummy data)
    # In reality, use your TextDataset/ClassifierDataLoader
    train_texts = ["sample text 1", "sample text 2"] * 1000
    train_labels = [0, 1] * 1000

    val_texts = ["validation text 1", "validation text 2"] * 100
    val_labels = [0, 1] * 100

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

    # 4. Configure training arguments
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),  # Mixed precision if CUDA available
        logging_steps=10,
        per_device_eval_batch_size=8,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        seed=42,
        remove_unused_columns=False,

        output_dir="./experiment_results",
        logging_dir="./logs",
    )

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
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
