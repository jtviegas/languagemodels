"""Dataset utilities for loading and processing text data from Hugging Face datasets.

This module provides classes and utilities for handling tokenized text datasets,
including the TextDataset class for creating PyTorch-compatible datasets
with automatic tokenization, padding, and truncation.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd


class TextDataset(Dataset):
    """A dataset class for tokenized text data.

    Attributes
    ----------
    data : Dataset
        The dataset containing the text and label data.
    text_col : str
        Column name for text data.
    label_col : str
        Column name for label data.
    encoded_texts : list
        List of tokenized and padded text sequences.
    max_length : int
        Maximum sequence length for padding/truncation.

    Methods
    -------
    __getitem__(index: int) -> tuple[torch.Tensor, torch.Tensor]
        Return encoded text and label as tensors for the given index.
    __len__() -> int
        Return the number of samples in the dataset.
    _longest_encoded_length() -> int
        Return the length of the longest encoded text.

    Methods
    -------
    __getitem__(index: int) -> tuple[torch.Tensor, torch.Tensor]
        Return encoded text and label as tensors for the given index.
    __len__() -> int
        Return the number of samples in the dataset.
    _longest_encoded_length() -> int
        Return the length of the longest encoded text.
    """

    def __init__(
        self,
        df: pd.DataFrame|Dataset,
        tokenizer,
        text_col: str = "text",
        label_col: str = "label",
        max_length=None,
        pad_token_id=50256,
    ) -> None:
        """Initialize the dataset with tokenized texts and labels.

        Args:
            df: DataFrame or Dataset containing the data.
            tokenizer: Tokenizer to encode text.
            text_col: Column name for text data.
            label_col: Column name for label data.
            max_length: Maximum sequence length. If None, uses the longest sequence.
            pad_token_id: Token ID to use for padding.
        """
        self.data = df
        self.text_col = text_col
        self.label_col = label_col

        if isinstance(df, pd.DataFrame):
            self.data = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

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
        label = self.data[self.label_col][index]
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """Return the length of the longest encoded text."""
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(max_length, len(encoded_text))
        return max_length
