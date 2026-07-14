"""Dataset utilities for loading and processing text data from Hugging Face datasets.

This module provides classes and utilities for handling tokenized text datasets,
including the TextDataset class for creating Hugging Face Dataset objects
with automatic tokenization, padding, and truncation.
"""

from datasets import Dataset as HFDataset


class TextDataset(HFDataset):
    """Dataset class for tokenized text with automatic padding and truncation.

    This class extends Hugging Face Dataset to handle text tokenization,
    padding, and truncation for machine learning models.
    """

    def __init__(
        self,
        tokenizer,
        texts: list[str],
        labels: list[int],
        max_length=None,
        pad_token_id=50256,
    ) -> None:
        """Initialize TextDataset with tokenized text and labels.

        Args:
            tokenizer: Tokenizer instance with encode method.
            texts: List of text strings to tokenize.
            labels: List of integer labels corresponding to texts.
            max_length: Maximum sequence length. If None, uses longest encoded text.
            pad_token_id: Token ID used for padding sequences (default: 50256).
        """
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

        hf_dataset = HFDataset.from_dict(
            {
                "input_ids": self.encoded_texts,
                "labels": self.labels,
            }
        )

        super().__init__(
            arrow_table=hf_dataset._data,  # noqa: SLF001
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=hf_dataset._indices,  # noqa: SLF001
            fingerprint=hf_dataset._fingerprint,  # noqa: SLF001
        )

    def _longest_encoded_length(self) -> int:
        """Return the length of the longest encoded text."""
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(max_length, len(encoded_text))
        return max_length
