"""Chunked WADI loader with file-local missing-value handling."""

import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


_FEATURE_COUNT = 127
_WADI_CHUNK_ROWS = 65_536
_METADATA_COLUMNS = {
    "row",
    "date",
    "time",
    "timestamp",
    "datetime",
}


def _read_header(path: Path) -> Tuple[list, list]:
    if not path.is_file():
        raise FileNotFoundError(f"Required WADI file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        original = next(csv.reader(handle), None)
    if original is None:
        raise ValueError(f"WADI CSV is empty: {path}")
    stripped = [str(column).strip() for column in original]
    if any(not column for column in stripped):
        raise ValueError(f"WADI CSV contains an empty column name: {path}")
    if len(stripped) != len(set(stripped)):
        raise ValueError(f"WADI CSV has duplicate columns after strip: {path}")
    return original, stripped


def _column_selection(
    original: Sequence[str],
    stripped: Sequence[str],
    *,
    require_attack: bool,
) -> Tuple[list, list, Optional[str]]:
    attack_indices = [
        index
        for index, column in enumerate(stripped)
        if column.casefold() == "attack"
    ]
    if require_attack and len(attack_indices) != 1:
        raise ValueError(
            "WADI attack CSV must contain exactly one Attack label column"
        )
    if not require_attack and attack_indices:
        raise ValueError("WADI normal CSV cannot contain an Attack label column")
    attack_index = attack_indices[0] if attack_indices else None
    feature_indices = [
        index
        for index, column in enumerate(stripped)
        if index != attack_index
        and column.casefold() not in _METADATA_COLUMNS
    ]
    if len(feature_indices) != _FEATURE_COUNT:
        raise ValueError(
            f"WADI requires exactly {_FEATURE_COUNT} sensor feature columns, "
            f"got {len(feature_indices)}"
        )
    feature_original = [original[index] for index in feature_indices]
    feature_stripped = [stripped[index] for index in feature_indices]
    attack_original = None if attack_index is None else original[attack_index]
    return feature_original, feature_stripped, attack_original


def _forward_fill_chunk(
    values: np.ndarray,
    previous: np.ndarray,
) -> None:
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        if np.isinf(column).any():
            raise ValueError("WADI sensor values must be finite or missing")
        valid = ~np.isnan(column)
        if not valid.any():
            if np.isfinite(previous[column_index]):
                column[:] = previous[column_index]
            continue
        first_valid = int(np.flatnonzero(valid)[0])
        if first_valid > 0 and np.isfinite(previous[column_index]):
            column[:first_valid] = previous[column_index]
        missing = np.isnan(column)
        if missing.any():
            indices = np.where(~missing, np.arange(column.size), 0)
            np.maximum.accumulate(indices, out=indices)
            column[:] = column[indices]
        finite_positions = np.flatnonzero(np.isfinite(column))
        if finite_positions.size:
            previous[column_index] = column[finite_positions[-1]]


def _load_file(
    path: Path,
    *,
    feature_original: Sequence[str],
    feature_names: Sequence[str],
    attack_original: Optional[str],
    allow_all_missing: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    usecols = list(feature_original)
    if attack_original is not None:
        usecols.append(attack_original)
    try:
        reader = pd.read_csv(
            path,
            usecols=usecols,
            chunksize=_WADI_CHUNK_ROWS,
        )
    except Exception as exc:
        raise ValueError(f"Cannot open WADI CSV in chunks: {path}") from exc

    value_chunks = []
    label_chunks = []
    previous = np.full(len(feature_original), np.nan, dtype=np.float32)
    for chunk_number, chunk in enumerate(reader, start=1):
        try:
            numeric = chunk.loc[:, feature_original].apply(
                pd.to_numeric,
                errors="raise",
            )
            values = numeric.to_numpy(dtype=np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"WADI sensor values must be numeric in {path.name} "
                f"chunk {chunk_number}"
            ) from exc
        _forward_fill_chunk(values, previous)
        value_chunks.append(values)

        if attack_original is not None:
            try:
                raw_labels = pd.to_numeric(
                    chunk[attack_original],
                    errors="raise",
                ).to_numpy(dtype=np.float32, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("WADI Attack labels must be numeric 1/-1") from exc
            if not np.isfinite(raw_labels).all():
                raise ValueError("WADI Attack labels must be finite")
            if not np.isin(raw_labels, (-1, 1)).all():
                raise ValueError(
                    "WADI Attack labels must be exactly 1 or -1"
                )
            label_chunks.append((raw_labels == -1).astype(np.int32))

    if not value_chunks:
        raise ValueError(f"WADI CSV contains no data rows: {path}")
    values = np.concatenate(value_chunks, axis=0).astype(np.float32, copy=False)
    for column_index, column_name in enumerate(feature_names):
        finite_positions = np.flatnonzero(np.isfinite(values[:, column_index]))
        if finite_positions.size == 0:
            if not allow_all_missing:
                raise ValueError(
                    f"WADI sensor column {column_name!r} is missing "
                    "for the entire file"
                )
            continue
        first_finite = int(finite_positions[0])
        if first_finite:
            values[:first_finite, column_index] = values[
                first_finite, column_index
            ]
    if not allow_all_missing and not np.isfinite(values).all():
        raise ValueError(f"WADI file still contains non-finite values: {path}")
    labels = (
        None
        if attack_original is None
        else np.concatenate(label_chunks, axis=0)
    )
    return values, labels


def load_data(root_path: str) -> LoadedTimeSeries:
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"WADI root directory not found: {root}")
    normal_path = root / "WADI_14days_new.csv"
    attack_path = root / "WADI_attackdataLABLE.csv"
    normal_original, normal_stripped = _read_header(normal_path)
    attack_original, attack_stripped = _read_header(attack_path)
    train_original, train_names, _ = _column_selection(
        normal_original,
        normal_stripped,
        require_attack=False,
    )
    test_original, test_names, attack_column = _column_selection(
        attack_original,
        attack_stripped,
        require_attack=True,
    )
    if train_names != test_names:
        raise ValueError(
            "WADI normal and attack sensor columns do not align after stripping "
            "column names"
        )

    train_values, _ = _load_file(
        normal_path,
        feature_original=train_original,
        feature_names=train_names,
        attack_original=None,
        allow_all_missing=True,
    )
    test_values, test_labels = _load_file(
        attack_path,
        feature_original=test_original,
        feature_names=test_names,
        attack_original=attack_column,
        allow_all_missing=True,
    )
    train_missing = np.isnan(train_values).all(axis=0)
    test_missing = np.isnan(test_values).all(axis=0)
    if not np.array_equal(train_missing, test_missing):
        missing_train = [
            name for name, missing in zip(train_names, train_missing) if missing
        ]
        missing_test = [
            name for name, missing in zip(test_names, test_missing) if missing
        ]
        raise ValueError(
            "WADI sensor columns missing for only one file: "
            f"train={missing_train}, test={missing_test}"
        )
    zero_filled_features = [
        name for name, missing in zip(train_names, train_missing) if missing
    ]
    if zero_filled_features:
        train_values[:, train_missing] = 0.0
        test_values[:, test_missing] = 0.0
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("WADI files still contain non-finite values")
    return LoadedTimeSeries(
        train_segments=[
            TimeSeriesSegment(
                "wadi",
                train_values,
                np.zeros(train_values.shape[0], dtype=np.int32),
            )
        ],
        test_segments=[
            TimeSeriesSegment("wadi", test_values, test_labels)
        ],
        metadata={"zero_filled_features": zero_filled_features},
    )
