"""Unit tests for the utils_data module."""

import datasets
import pandas as pd
import torch

from tgedr_languagemodels.utils.utils_data import ClassifierDataLoader, TextDataset


class DummyTokenizer:
    """Simple tokenizer stub with deterministic encodings."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str) -> list[int]:
        return self.mapping[text]


class TestTextDataset:
    """Test suite for TextDataset."""

    def test_init_without_max_length_computes_longest_and_pads(self) -> None:
        """TextDataset computes max length from inputs and pads shorter sequences."""
        df = pd.DataFrame(
            {
                "text": ["a", "b"],
                "label": [1, 0],
            },
        )
        tokenizer = DummyTokenizer({"a": [10, 11], "b": [12]})

        dataset = TextDataset(df, tokenizer, pad_token_id=99)

        assert dataset.max_length == 2
        assert dataset.encoded_texts == [[10, 11], [12, 99]]

    def test_init_with_max_length_truncates_and_pads_custom_columns(self) -> None:
        """TextDataset truncates when max_length is provided and supports custom columns."""
        df = pd.DataFrame(
            {
                "content": ["short", "long"],
                "target": [3, 4],
            },
        )
        tokenizer = DummyTokenizer({"short": [1], "long": [2, 3, 4, 5]})

        dataset = TextDataset(
            df,
            tokenizer,
            text_col="content",
            label_col="target",
            max_length=3,
            pad_token_id=0,
        )

        assert dataset.max_length == 3
        assert dataset.encoded_texts == [[1, 0, 0], [2, 3, 4]]

    def test_getitem_and_len_return_expected_tensors(self) -> None:
        """TextDataset returns long tensors and length equals number of rows."""
        df = pd.DataFrame(
            {
                "text": ["x", "y", "z"],
                "label": [7, 8, 9],
            },
        )
        tokenizer = DummyTokenizer({"x": [1], "y": [2, 3], "z": [4]})

        dataset = TextDataset(df, tokenizer, pad_token_id=5)
        encoded, label = dataset[1]

        assert len(dataset) == 3
        assert torch.equal(encoded, torch.tensor([2, 3], dtype=torch.long))
        assert torch.equal(label, torch.tensor(8, dtype=torch.long))
        assert encoded.dtype == torch.long
        assert label.dtype == torch.long

    def test_longest_encoded_length_with_empty_input_is_zero(self) -> None:
        """Internal longest-length helper returns 0 for empty datasets."""
        empty_df = pd.DataFrame({"text": [], "label": []})
        tokenizer = DummyTokenizer({})

        dataset = TextDataset(empty_df, tokenizer)

        assert dataset._longest_encoded_length() == 0
        assert dataset.max_length == 0


class TestClassifierDataLoader:
    """Test suite for ClassifierDataLoader."""

    def _build_dataframe(self, n_per_class: int = 10) -> pd.DataFrame:
        rows = []
        for i in range(n_per_class):
            rows.append({"text": f"pos-{i}", "label": 1, "label_text": "pos"})
            rows.append({"text": f"neg-{i}", "label": 0, "label_text": "neg"})
        return pd.DataFrame(rows)

    def test_setup_class_label_casts_column(self) -> None:
        """_setup_class_label should cast label_text to ClassLabel feature."""
        df = self._build_dataframe(n_per_class=3)
        ds = datasets.Dataset.from_pandas(df)
        loader = ClassifierDataLoader(tokenizer=DummyTokenizer({}), batch_size=2)

        cast_ds = loader._setup_class_label(ds)

        assert str(cast_ds.features["label_text"]) == "ClassLabel(names=['neg', 'pos'])"

    def test_create_returns_train_validation_test_loaders(self) -> None:
        """create should return all three loaders from a pandas dataframe."""
        df = self._build_dataframe(n_per_class=10)
        tok_map = {row["text"]: [idx, idx + 1] for idx, row in enumerate(df.to_dict("records"))}
        tokenizer = DummyTokenizer(tok_map)

        loader_factory = ClassifierDataLoader(
            tokenizer=tokenizer,
            batch_size=4,
            validation_split=0.2,
            test_split=0.2,
        )
        splits = loader_factory.create(df)

        assert set(splits.keys()) == {"train", "validation", "test"}
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

        batch_inputs, batch_targets = next(iter(splits["train"]))
        assert batch_inputs.dtype == torch.long
        assert batch_targets.dtype == torch.long
