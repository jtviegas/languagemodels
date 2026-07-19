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
from tgedr_lm.commons.utils_io import load_pickle_compressed

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
        self._weights = None
        self._weights_url = None

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

    def _get_model(self) -> GPT2Classifier:
        model = GPT2Classifier(ClassifierBaseConfiguration(n_classes=3))
        if self._weights is not None:
            model.pretrain(self._weights)
        return model

    def _search_hyperparameters(self) -> dict[str, Any]:
        train_dataset, val_dataset = self.get_datasets(fraction=0.3)
        args = TrainingArgs()
        args.set("output_dir", str(OUTPUT_DIR))
        hp_search = HyperParamSearch(GPT2Classifier.compute_metrics, train_args=args.to_training_arguments())
        hyperparameters = hp_search.search(
            model=self._get_model(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            trials=3,
        )
        return hyperparameters

    @Etl.inject_configuration
    def extract(self, dataset: str, weights_url: str | None = None) -> Any:
        """Extract data from source dataset.

        Parameters
        ----------
        dataset : str
            Name of the dataset to extract.
        weights_url : str | None, optional
            URL to the pretrained weights, by default None

        Returns
        -------
        Any
            None
        weights_url : str
            URL to the pretrained weights.
        """
        self._fetch_data(dataset)

        if weights_url is not None:
            self._weights_url = Path(weights_url)
            self._weights = load_pickle_compressed(weights_url)

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

        model = self._get_model()
        train_dataset, val_dataset = self.get_datasets(fraction=1.0)
        self._trainer = Trainer(
            model=model,
            args=args.to_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=model.compute_metrics,
        )
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
        if self._weights_url is not None:
            shutil.copy2(self._weights_url, OUTPUT_DIR / self._weights_url.name)
        self._trainer.push_to_hub(commit_message="Training completed!")
