"""I/O utilities for loading and saving data in various formats.

This module provides utility functions for loading dictionaries from JSON or pickle files,
with optional gzip compression support.
"""

import json
import gzip
import pickle  # nosec B403 - Used for trusted internal serialization, not untrusted input
from pathlib import Path


def _is_pickle_path(path: Path) -> bool:
    """Return True if the path has a pickle or compressed-pickle suffix.

    Parameters
    ----------
    path : Path
        File path to inspect.

    Returns
    -------
    bool
        True when the path ends in .pickle, .pkl, or .pickle.gz / .pkl.gz.
    """
    suffixes = path.suffixes
    if suffixes[-1] == ".gz" and len(suffixes) > 1:
        return suffixes[-2] in {".pickle", ".pkl"}
    return suffixes[-1] in {".pickle", ".pkl"}


def load_dict(source_path, *, use_pickle: bool | None = None, compress: bool = False) -> dict:
    """Deserialize a dictionary from a JSON or pickle file, with optional gzip decompression.

    Parameters
    ----------
    source_path : str or Path
        Source file path.
    use_pickle : bool or None, optional
        Force pickle format; inferred from file extension when None (default: None).
    compress : bool, optional
        Whether the file is gzip-compressed (default: False).

    Returns
    -------
    dict
        Deserialized data.
    """
    path = Path(source_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 # nosec B301

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_dict(data, target_path, *, use_pickle: bool | None = None, compress: bool = False) -> None:
    """Serialize a dictionary to disk as JSON or pickle, with optional gzip compression.

    Parameters
    ----------
    data : dict
        Data to serialize.
    target_path : str or Path
        Destination file path.
    use_pickle : bool or None, optional
        Force pickle format; inferred from file extension when None (default: None).
    compress : bool, optional
        Whether to apply gzip compression (default: False).
    """
    path = Path(target_path)
    if use_pickle is None:
        use_pickle = _is_pickle_path(path)

    if use_pickle:
        opener = gzip.open if compress or path.suffix == ".gz" else open
        with opener(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_pickle_compressed(source_path) -> dict:
    """Load Python data saved with pickle and gzip compression."""
    return load_dict(source_path, use_pickle=True, compress=True)


def save_pickle_compressed(data, target_path) -> None:
    """Save Python data using pickle and gzip compression."""
    save_dict(data, target_path, use_pickle=True, compress=True)
