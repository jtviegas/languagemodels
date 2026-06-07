"""Unit tests for the utils_llm module."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
from tgedr_languagemodels.utils.utils_llm import (
    text_to_token_ids,
    token_ids_to_text,
    generate,
    _is_pickle_path,
    save_dict,
    load_dict,
    save_pickle_compressed,
    load_pickle_compressed,
    longest_encoded_length,
    harmonize_text_sequences,
)


class TestTextTokenization:
    """Test suite for text tokenization functions."""

    def test_text_to_token_ids_shape(self) -> None:
        """Test text_to_token_ids output shape."""
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        result = text_to_token_ids("hello world", tokenizer)
        
        assert result.shape == (1, 5)
        assert isinstance(result, torch.Tensor)

    def test_text_to_token_ids_dtype(self) -> None:
        """Test text_to_token_ids dtype."""
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        
        result = text_to_token_ids("test", tokenizer)
        
        assert result.dtype == torch.long

    def test_token_ids_to_text_basic(self) -> None:
        """Test token_ids_to_text conversion."""
        tokenizer = Mock()
        tokenizer.decode.return_value = "hello world"
        
        token_ids = torch.tensor([[1, 2, 3]])
        result = token_ids_to_text(token_ids, tokenizer)
        
        assert isinstance(result, str)
        assert result == "hello world"

    def test_token_ids_to_text_squeeze(self) -> None:
        """Test that token_ids_to_text squeezes batch dimension."""
        tokenizer = Mock()
        tokenizer.decode.return_value = "test"
        
        token_ids = torch.tensor([[1, 2, 3]])
        token_ids_to_text(token_ids, tokenizer)
        
        # Verify that decode was called with a 1D list
        call_args = tokenizer.decode.call_args[0][0]
        assert isinstance(call_args, list)


class TestPicklePath:
    """Test suite for pickle path detection."""

    def test_pickle_extension(self) -> None:
        """Test detection of .pickle extension."""
        path = Path("model.pickle")
        assert _is_pickle_path(path) is True

    def test_pkl_extension(self) -> None:
        """Test detection of .pkl extension."""
        path = Path("model.pkl")
        assert _is_pickle_path(path) is True

    def test_pickle_gz_extension(self) -> None:
        """Test detection of .pickle.gz extension."""
        path = Path("model.pickle.gz")
        assert _is_pickle_path(path) is True

    def test_pkl_gz_extension(self) -> None:
        """Test detection of .pkl.gz extension."""
        path = Path("model.pkl.gz")
        assert _is_pickle_path(path) is True

    def test_json_extension(self) -> None:
        """Test that .json is not detected as pickle."""
        path = Path("data.json")
        assert _is_pickle_path(path) is False

    def test_txt_extension(self) -> None:
        """Test that .txt is not detected as pickle."""
        path = Path("data.txt")
        assert _is_pickle_path(path) is False

    def test_gz_only_extension(self) -> None:
        """Test that .gz alone is not detected as pickle."""
        path = Path("data.gz")
        assert _is_pickle_path(path) is False


class TestDictSerialization:
    """Test suite for dictionary save/load."""

    def test_save_and_load_json(self) -> None:
        """Test saving and loading JSON dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key1": "value1", "key2": 42}
            
            save_dict(data, path, use_pickle=False)
            loaded = load_dict(path, use_pickle=False)
            
            assert loaded == data

    def test_save_and_load_pickle(self) -> None:
        """Test saving and loading pickle dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            data = {"key1": "value1", "key2": [1, 2, 3]}
            
            save_dict(data, path, use_pickle=True)
            loaded = load_dict(path, use_pickle=True)
            
            assert loaded == data

    def test_save_pickle_compressed(self) -> None:
        """Test saving pickle with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl.gz"
            data = {"key": "value"}
            
            save_pickle_compressed(data, path)
            assert path.exists()

    def test_load_pickle_compressed(self) -> None:
        """Test loading pickle with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl.gz"
            data = {"key": "value"}
            
            save_pickle_compressed(data, path)
            loaded = load_pickle_compressed(path)
            
            assert loaded == data

    def test_infer_format_from_extension(self) -> None:
        """Test format inference from file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test pickle inference
            pkl_path = Path(tmpdir) / "test.pkl"
            data = {"test": "data"}
            save_dict(data, pkl_path)
            loaded = load_dict(pkl_path)
            assert loaded == data


class TestEncodedLengthFunctions:
    """Test suite for encoded text length functions."""

    def test_longest_encoded_length_basic(self) -> None:
        """Test longest_encoded_length basic functionality."""
        encoded_texts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        
        result = longest_encoded_length(encoded_texts)
        
        assert result == 4

    def test_longest_encoded_length_single_text(self) -> None:
        """Test longest_encoded_length with single text."""
        encoded_texts = [[1, 2, 3, 4, 5]]
        
        result = longest_encoded_length(encoded_texts)
        
        assert result == 5

    def test_longest_encoded_length_empty_list(self) -> None:
        """Test longest_encoded_length with empty list."""
        encoded_texts = []
        
        result = longest_encoded_length(encoded_texts)
        
        assert result == 0

    def test_longest_encoded_length_all_same_length(self) -> None:
        """Test longest_encoded_length when all texts have same length."""
        encoded_texts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        result = longest_encoded_length(encoded_texts)
        
        assert result == 3


class TestHarmonizeTextSequences:
    """Test suite for text sequence harmonization."""

    def test_harmonize_basic(self) -> None:
        """Test basic text sequence harmonization."""
        df = pd.DataFrame({
            "text": ["hello", "hi there", "goodbye"]
        })
        
        tokenizer = Mock()
        tokenizer.encode.side_effect = [
            [1, 2],
            [3, 4, 5],
            [6, 7, 8, 9]
        ]
        tokenizer.eot_token = 0
        
        result = harmonize_text_sequences(df, tokenizer)
        
        # All sequences should have length 4
        assert all(len(seq) == 4 for seq in result)

    def test_harmonize_with_sequence_length(self) -> None:
        """Test harmonize with explicit sequence length."""
        df = pd.DataFrame({
            "text": ["hello", "world"]
        })
        
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        tokenizer.eot_token = 0
        
        result = harmonize_text_sequences(df, tokenizer, sequence_length=5)
        
        assert all(len(seq) == 5 for seq in result)

    def test_harmonize_padding(self) -> None:
        """Test that harmonize pads with EOT token."""
        df = pd.DataFrame({
            "text": ["short", "longer text"]
        })
        
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1], [2, 3]]
        tokenizer.eot_token = 99
        
        result = harmonize_text_sequences(df, tokenizer)
        
        # Check that shorter sequence is padded
        assert 99 in result[0]
        assert result[0].count(99) > result[1].count(99)

    def test_harmonize_custom_column(self) -> None:
        """Test harmonize with custom text column."""
        df = pd.DataFrame({
            "content": ["test1", "test2"],
            "other": ["x", "y"]
        })
        
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1], [2, 3]]
        tokenizer.eot_token = 0
        
        result = harmonize_text_sequences(df, tokenizer, text_col="content")
        
        # Should process "content" column
        assert len(result) == 2
