"""GPT2 Classifier ETL pipeline for data extraction, transformation, and loading."""

import logging
from typing import Any
from pathlib import Path
import shutil
import tempfile
from datasets import load_dataset
import tiktoken
from transformers import Trainer
from tgedr_lm.classifier.text_dataset import TextDataset
from tgedr_dataops_abs.etl import Etl
from tgedr_lm.classifier.gpt2.hyperparam_search import HyperParamSearch
from tgedr_lm.classifier.gpt2.model import GPT2Classifier
from tgedr_lm.configuration import ClassifierBaseConfiguration, TrainingArgs

MODEL_CARD: Path = Path(__file__).resolve().parent / "README.md"
OUTPUT_DIR: Path = Path(tempfile.gettempdir()) / "gpt2classifier"

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class Gpt2ClassifierEtl(Etl):
    """GPT2 Classifier ETL pipeline for data extraction, transformation, and loading."""

    def __init__(self, config: dict) -> None:
        """Initialize Gpt2ClassifierEtl.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        super().__init__(config)
        self._data = {}
        self._trainer = None

    def _fetch_data(self, dataset: str) -> Any:
        """Fetch data from source.

        Returns
        -------
        Any
            Fetched data.
        """
        if self._data == {}:
            train_ds = load_dataset(dataset, split="train")
            ds = train_ds.train_test_split(test_size=0.2)
            self._data["train"] = {"texts": list(ds["train"]["sentence"]), "labels": list(ds["train"]["label"])}
            self._data["test"] = {"texts": list(ds["test"]["sentence"]), "labels": list(ds["test"]["label"])}

    def get_datasets(self, fraction: float = 1.0) -> tuple[TextDataset, TextDataset]:
        """Get train and validation datasets.

        Parameters
        ----------
        fraction : float, optional
            Fraction of data to use, by default 1.0

        Returns
        -------
        tuple[TextDataset, TextDataset]
            Training and validation datasets.
        """
        train_len = int(len(self._data["train"]["texts"]) * fraction)
        test_len = int(len(self._data["test"]["texts"]) * fraction)
        tokenizer = tiktoken.get_encoding("gpt2")
        train_texts = self._data["train"]["texts"][:train_len]
        train_labels = self._data["train"]["labels"][:train_len]
        test_texts = self._data["test"]["texts"][:test_len]
        test_labels = self._data["test"]["labels"][:test_len]
        train_dataset = TextDataset(tokenizer=tokenizer, texts=train_texts, labels=train_labels)
        val_dataset = TextDataset(tokenizer=tokenizer, texts=test_texts, labels=test_labels)
        return train_dataset, val_dataset

    def _search_hyperparameters(self) -> dict[str, Any]:
        train_dataset, val_dataset = self.get_datasets(fraction=0.3)
        args = TrainingArgs()
        args.set("output_dir", str(OUTPUT_DIR))
        hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=args.to_training_arguments())
        hyperparameters = hp_search.search(
            model=GPT2Classifier(ClassifierBaseConfiguration(n_classes=3)),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            trials=3,
        )
        return hyperparameters

    @Etl.inject_configuration
    def extract(self, dataset: str) -> Any:
        """Extract data from source dataset.

        Parameters
        ----------
        dataset : str
            Name of the dataset to extract.

        Returns
        -------
        Any
            None
        """
        self._fetch_data(dataset)

    def transform(self) -> Any:
        """Transform data by searching hyperparameters and training the model.

        Returns
        -------
        Any
            None
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        hyperparameters = self._search_hyperparameters()
        args = TrainingArgs()
        for key in [
            "learning_rate",
            "weight_decay",
            "num_train_epochs",
            "per_device_train_batch_size",
            "warmup_steps",
        ]:
            if key in hyperparameters:
                args.set(key, hyperparameters[key])
        args.set("hub_model_id", "jtviegas/gpt2classifier")
        args.set("output_dir", str(OUTPUT_DIR))

        model = GPT2Classifier(ClassifierBaseConfiguration(n_classes=3))
        train_dataset, val_dataset = self.get_datasets(fraction=1.0)
        self._trainer = Trainer(
            model=model,
            args=args.to_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=model.compute_metrics,
        )
        shutil.copy2(MODEL_CARD, OUTPUT_DIR / MODEL_CARD.name)
        self._trainer.train()
        final_metrics = self._trainer.evaluate()
        logger.info(f"Final evaluation metrics: {final_metrics}")

    def load(self) -> Any:
        """Load transformed data to destination.

        Returns
        -------
        Any
            Result of load operation.
        """
        self._trainer.push_to_hub(commit_message="Training completed!")
