"""Example: Using GPT2Classifier with Hugging Face Trainer.

This example demonstrates how to train GPT2Classifier using the Hugging Face
Trainer API, which provides features like distributed training, gradient
accumulation, mixed precision, and comprehensive logging.
"""

from tgedr_languagemodels import create_trainer_ready_classifier
from tgedr_languagemodels.configuration import BaseClassifierConfig
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


class SimpleTextDataset(Dataset):
    """Simple example dataset for demonstration.

    In practice, you would use your actual TextDataset or ClassifierDataLoader.
    """

    def __init__(self, texts, labels, tokenizer, max_length=1024):
        """Initialize dataset.

        Parameters
        ----------
        texts : list of str
            Text samples.
        labels : list of int
            Class labels.
        tokenizer : function
            Function to convert text to token IDs.
        max_length : int
            Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single sample.

        Returns
        -------
        dict
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and pad
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def example_trainer_usage():
    """Example: Train GPT2Classifier using Hugging Face Trainer.

    This example shows the basic setup. For production use:
    1. Use your actual datasets (TextDataset, ClassifierDataLoader)
    2. Configure TrainingArguments for your hardware
    3. Add custom callbacks for logging/monitoring
    4. Use a proper data collator if needed
    """

    # 1. Configure the model
    config = BaseClassifierConfig(
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
    model = create_trainer_ready_classifier(config)

    # 3. Prepare datasets (example with dummy data)
    # In reality, use your TextDataset/ClassifierDataLoader
    train_texts = ["sample text 1", "sample text 2"] * 10
    train_labels = [0, 1] * 10

    val_texts = ["validation text 1", "validation text 2"] * 5
    val_labels = [0, 1] * 5

    # Simple tokenizer for demo (use proper tokenizer in practice)
    def dummy_tokenizer(text, **kwargs):
        # Just return dummy token IDs for this example
        token_ids = [1, 2, 3, 4, 5]
        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.ones(len(token_ids)),
        }

    train_dataset = SimpleTextDataset(train_texts, train_labels, dummy_tokenizer)
    val_dataset = SimpleTextDataset(val_texts, val_labels, dummy_tokenizer)

    # 4. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_classifier_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=50,
        evaluation_strategy="steps",
        save_steps=50,
        save_strategy="steps",
        learning_rate=3e-5,
        fp16=torch.cuda.is_available(),  # Mixed precision if CUDA available
        gradient_accumulation_steps=1,
        seed=42,
    )

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # Optional: add callbacks for custom behavior
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save model
    model.save_pretrained("./final_model")

    print("Training complete!")

    return trainer


def example_with_your_dataloader():
    """Example: Using your existing ClassifierDataLoader with Trainer.

    This shows how to adapt your existing data loading infrastructure.
    """
    from tgedr_languagemodels.utils.utils_data import ClassifierDataLoader

    # Your existing data setup
    config = BaseClassifierConfig(
        vocabulary_size=50257,
        embeddings_dimension=768,
        context_length=1024,
        n_layers=12,
        drop_rate=0.1,
        stride=1,
        n_heads=12,
        n_classes=2,
    )

    model = create_trainer_ready_classifier(config)

    # Create your existing DataLoader
    loader = ClassifierDataLoader(
        df=None,  # your dataframe
        text_column="text",
        label_column="label",
        test_size=0.2,
        val_size=0.1,
    )

    # Note: You may need to create a Dataset wrapper around your DataLoader
    # or refactor to use Hugging Face datasets library

    # This is just a conceptual example
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,  # Adapt from loader
    #     eval_dataset=eval_dataset,    # Adapt from loader
    # )
    # trainer.train()


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
