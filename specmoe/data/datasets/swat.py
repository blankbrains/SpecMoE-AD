"""Strict single-segment SWaT dataset loader."""

from pathlib import Path

import numpy as np
import pandas as pd

from specmoe.data.datasets._io import first_row_is_header
from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


_FEATURE_COUNT = 51
_LABEL_ALIASES = {"attack", "label", "anomaly", "class"}
_METADATA_COLUMNS = {"row", "date", "time", "timestamp", "datetime"}


def _resolve_path(root: Path, primary: str, alternative: str) -> Path:
    primary_path = root / primary
    if primary_path.is_file():
        return primary_path
    alternative_path = root / alternative
    if alternative_path.is_file():
        return alternative_path
    raise FileNotFoundError(f"SWaT data file not found: {primary_path}")


def _read_frame(path: Path) -> pd.DataFrame:
    has_header = first_row_is_header(path)
    try:
        frame = pd.read_csv(path, header=0 if has_header else None)
    except Exception as exc:
        raise ValueError(f"Cannot parse SWaT CSV: {path}") from exc
    if frame.empty:
        raise ValueError(f"SWaT CSV contains no data rows: {path}")
    if has_header:
        columns = [str(column).strip() for column in frame.columns]
        if any(not column for column in columns) or len(columns) != len(set(columns)):
            raise ValueError("SWaT columns must be non-empty and unique after strip")
        frame.columns = columns
    else:
        if frame.shape[1] != _FEATURE_COUNT + 1:
            raise ValueError(
                f"Headerless SWaT data must have {_FEATURE_COUNT + 1} columns, "
                f"got {frame.shape[1]}"
            )
        frame.columns = ["attack"] + [
            f"feature_{index}" for index in range(_FEATURE_COUNT)
        ]
    return frame


def _extract(frame: pd.DataFrame, *, source: str):
    label_columns = [
        column
        for column in frame.columns
        if str(column).strip().casefold() in _LABEL_ALIASES
    ]
    if len(label_columns) != 1:
        raise ValueError(
            f"SWaT {source} data must contain exactly one binary label column"
        )
    label_column = label_columns[0]
    feature_columns = [
        column
        for column in frame.columns
        if column != label_column
        and str(column).strip().casefold() not in _METADATA_COLUMNS
    ]
    if len(feature_columns) != _FEATURE_COUNT:
        raise ValueError(
            f"SWaT {source} data must have {_FEATURE_COUNT} feature columns, "
            f"got {len(feature_columns)}"
        )
    try:
        values = frame.loc[:, feature_columns].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
        labels = pd.to_numeric(
            frame[label_column], errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SWaT {source} features and labels must be numeric") from exc
    return feature_columns, values, labels


def load_data(root_path: str) -> LoadedTimeSeries:
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"SWaT root directory not found: {root}")
    train_path = _resolve_path(
        root, "SWaT_Dataset_Normal_v1_train.csv", "swat_train.csv"
    )
    test_path = _resolve_path(
        root, "SWaT_Dataset_Attack_v0_test.csv", "swat_test.csv"
    )
    train_columns, train_values, train_labels = _extract(
        _read_frame(train_path), source="train"
    )
    test_columns, test_values, test_labels = _extract(
        _read_frame(test_path), source="test"
    )
    if train_columns != test_columns:
        raise ValueError("SWaT train/test feature columns do not align")
    train_segment = TimeSeriesSegment("swat", train_values, train_labels)
    if np.any(train_segment.labels != 0):
        raise ValueError("SWaT normal training file contains attack labels")
    return LoadedTimeSeries(
        train_segments=[train_segment],
        test_segments=[TimeSeriesSegment("swat", test_values, test_labels)],
    )
