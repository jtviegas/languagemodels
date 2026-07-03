"""Dataset utilities for loading and processing text data from Hugging Face datasets.

This module provides classes and utilities for handling tokenized text datasets,
including the TextDataset class for creating PyTorch-compatible datasets
with automatic tokenization, padding, and truncation.
"""
import datasets
import torch
import pandas as pd
from torch.utils.data import Dataset as TorchDataset


class TextDataset(datasets.Dataset, TorchDataset):

    def __init__(
        self,
        tokenizer,
        texts: list[str],
        labels: list[int],
        max_length=None,
        pad_token_id=50256,
    ) -> None:

        self.texts = texts
        self.labels = labels

        # 1 Pretokenizes texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.texts]

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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, index: int | slice | list) -> dict:
        """Return one sample or a batch following Hugging Face Dataset conventions."""
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return {
                "input_ids": [torch.tensor(self.encoded_texts[i], dtype=torch.long) for i in indices],
                "labels": [torch.tensor(self.labels[i], dtype=torch.long) for i in indices],
            }

        if isinstance(index, list):
            return {
                "input_ids": [torch.tensor(self.encoded_texts[i], dtype=torch.long) for i in index],
                "labels": [torch.tensor(self.labels[i], dtype=torch.long) for i in index],
            }

        encoded = self.encoded_texts[index]
        label = self.labels[index]
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _longest_encoded_length(self) -> int:
        """Return the length of the longest encoded text."""
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(max_length, len(encoded_text))
        return max_length
