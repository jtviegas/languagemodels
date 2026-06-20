"""Utilities for handling text datasets in PyTorch.

This module provides classes and functions for preparing text data for machine learning models.
It includes TextDataset, a PyTorch Dataset subclass that handles tokenization, padding, and
conversion of text data to tensors for model training and inference.
"""

import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class TextDataset(Dataset):
    """A PyTorch Dataset for text classification with tokenized and padded text sequences.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing the text and label data.
    text_col : str
        Name of the column containing text data.
    label_col : str
        Name of the column containing label data.
    max_length : int
        Maximum sequence length for tokenized texts.
    encoded_texts : list
        List of tokenized and padded text sequences.

    Methods
    -------
    __getitem__(index) -> tuple[torch.Tensor, torch.Tensor]
        Return encoded text and label as tensors for the given index.
    __len__() -> int
        Return the number of samples in the dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "text",
        label_col: str = "label",
        max_length=None,
        pad_token_id=50256,
    ) -> None:
        """Initialize the TextDataset with tokenized and padded text sequences.

        Args:
            df: DataFrame containing the text and label data.
            tokenizer: Tokenizer object with an encode method.
            text_col: Name of the column containing text data.
            label_col: Name of the column containing label data.
            max_length: Maximum sequence length; if None, uses the longest sequence.
            pad_token_id: Token ID used for padding sequences.
        """
        self.data = df
        self.text_col = text_col
        self.label_col = label_col

        # 1 Pretokenizes texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data[self.text_col]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 2 Truncates sequences if they are longer than max_length
            self.encoded_texts = [encoded_text[: self.max_length] for encoded_text in self.encoded_texts]

        # 3 Pads sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Return encoded text and label as tensors for the given index."""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index][self.label_col]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """Return the length of the longest encoded text."""
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(max_length, len(encoded_text))
        return max_length


class ClassifierDataLoader:
    """Factory for creating DataLoader instances with consistent parameters.

    Attributes
    ----------
    text_col : str
        Name of the column containing text data.
    label_col : str
        Name of the column containing label data.
    tokenizer : object
        Tokenizer object to use for encoding text data.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the data at every epoch.
    num_workers : int
        Number of subprocesses to use for data loading.
    validation_split : float
        Fraction of data to use for validation.
    test_split : float
        Fraction of data to use for testing.

    Methods
    -------
    create(data) -> dict[str, DataLoader]
        Create train, validation, and test DataLoaders from input data.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        shuffle: bool = True,  # noqa: FBT001, FBT002
        num_workers: int = 0,
        validation_split: float = 0.2,
        test_split: float = 0.2,
        text_col: str = "text",
        label_col: str = "label",
    ) -> None:
        """Factory for creating DataLoader instances with consistent parameters.

        Args:
            text_col: Name of the column containing text data.
            label_col: Name of the column containing label data.
            tokenizer: Tokenizer object to use for encoding text data.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data at every epoch.
            num_workers: Number of subprocesses to use for data loading.
            validation_split: Fraction of data to use for validation.
            test_split: Fraction of data to use for testing.

        """
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.test_split = test_split

    def create(self, data: pd.DataFrame | datasets.Dataset) -> dict[str, DataLoader]:
        """Create train, validation, and test DataLoaders from the input dataset.

        Args:
            data: Input data as a pandas DataFrame or Hugging Face Dataset.

        Returns:
            Dictionary containing `train`, `val`, and `test` DataLoaders.
        """
        ds = datasets.Dataset.from_pandas(data) if isinstance(data, pd.DataFrame) else data
        ds = self._setup_class_label(ds)
        ds_split = ds.train_test_split(
            test_size=(self.validation_split + self.test_split), seed=42, stratify_by_column="label_text"
        )
        df_train = ds_split["train"].to_pandas()[[self.text_col, self.label_col]]
        ds_split = ds_split["test"].train_test_split(
            test_size=self.test_split / (self.validation_split + self.test_split),
            seed=42,
            stratify_by_column="label_text",
        )
        df_val = ds_split["train"].to_pandas()[[self.text_col, self.label_col]]
        df_test = ds_split["test"].to_pandas()[[self.text_col, self.label_col]]

        train_ds = TextDataset(df_train, self.tokenizer, text_col=self.text_col, label_col=self.label_col)
        val_ds = TextDataset(df_val, self.tokenizer, text_col=self.text_col, label_col=self.label_col)
        test_ds = TextDataset(df_test, self.tokenizer, text_col=self.text_col, label_col=self.label_col)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False
        )

        return {"train": train_loader, "validation": val_loader, "test": test_loader}

    def _setup_class_label(self, ds: Dataset, label_column: str = "label_text") -> Dataset:
        """Set up class labels for a dataset by converting a column to ClassLabel type.

        Args:
            ds: Dataset containing the label column to convert.
            label_column: Name of the column containing labels to convert.

        Returns:
            Dataset with the specified column converted to ClassLabel type.
        """
        unique_labels = sorted(set(ds[label_column]))
        ds = ds.cast_column(label_column, datasets.ClassLabel(names=[str(x) for x in unique_labels]))
        return ds
