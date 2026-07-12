"""Unit tests for TextDataset branch coverage."""

from tgedr_lm.classifier.text_dataset import TextDataset


class _FakeTokenizer:
    """Simple tokenizer stub with deterministic token lengths."""

    @staticmethod
    def encode(text: str) -> list[int]:
        mapping = {
            "a": [1, 2, 3, 4, 5],
            "b": [6, 7],
        }
        return mapping[text]


def test_text_dataset_truncates_when_max_length_is_provided() -> None:
    dataset = TextDataset(
        tokenizer=_FakeTokenizer(),
        texts=["a", "b"],
        labels=[1, 0],
        max_length=3,
        pad_token_id=99,
    )

    # "a" is truncated from length 5 to 3.
    assert dataset[0]["input_ids"] == [1, 2, 3]
    # "b" is padded up to max_length after truncation logic.
    assert dataset[1]["input_ids"] == [6, 7, 99]
    assert dataset[0]["labels"] == 1
    assert dataset[1]["labels"] == 0
