"""Unit tests for utils_io helpers."""

import gzip
import json
import pickle
from pathlib import Path

from tgedr_lm.commons.utils_io import _is_pickle_path, load_dict, load_pickle_compressed, save_dict, save_pickle_compressed


def test_is_pickle_path_detects_supported_suffixes() -> None:
    assert _is_pickle_path(Path("weights.pickle")) is True
    assert _is_pickle_path(Path("weights.pkl")) is True
    assert _is_pickle_path(Path("weights.pickle.gz")) is True
    assert _is_pickle_path(Path("weights.pkl.gz")) is True
    assert _is_pickle_path(Path("weights.json")) is False


def test_load_dict_loads_json_when_pickle_not_requested(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    payload = {"a": 1, "b": "two"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_dict(path) == payload


def test_load_dict_loads_plain_pickle(tmp_path: Path) -> None:
    path = tmp_path / "data.pickle"
    payload = {"a": 1, "b": [1, 2, 3]}
    with path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)

    assert load_dict(path) == payload


def test_load_dict_loads_gzip_pickle_by_suffix_inference(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl.gz"
    payload = {"compressed": True}
    with gzip.open(path, "wb") as file_obj:
        pickle.dump(payload, file_obj)

    assert load_dict(path) == payload


def test_load_pickle_compressed_forces_compressed_pickle_loading(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    payload = {"weights": [1, 2, 3]}
    with gzip.open(path, "wb") as file_obj:
        pickle.dump(payload, file_obj)

    assert load_pickle_compressed(path) == payload


def test_save_dict_writes_json_when_pickle_not_requested(tmp_path: Path) -> None:
    path = tmp_path / "saved.json"
    payload = {"a": 10, "b": "hello"}

    save_dict(payload, path)

    assert load_dict(path) == payload


def test_save_dict_writes_plain_pickle_when_extension_is_pickle(tmp_path: Path) -> None:
    path = tmp_path / "saved.pickle"
    payload = {"x": [1, 2], "y": 3}

    save_dict(payload, path)

    assert load_dict(path) == payload


def test_save_dict_writes_gzip_pickle_when_compressed(tmp_path: Path) -> None:
    path = tmp_path / "saved.pkl.gz"
    payload = {"compressed": True, "v": 42}

    save_dict(payload, path)

    assert load_dict(path) == payload


def test_save_pickle_compressed_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    payload = {"weights": [7, 8, 9]}

    save_pickle_compressed(payload, path)

    assert load_pickle_compressed(path) == payload
