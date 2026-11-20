"""Shared strict loader for the NASA MSL and SMAP benchmarks."""

import ast
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


def _load_entity(path: Path, expected_features: int, dataset_name: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load {dataset_name} entity file {path}") from exc
    if values.ndim != 2:
        raise ValueError(
            f"{dataset_name} entity {path.name} must be two-dimensional, "
            f"got {values.shape}"
        )
    if values.shape[1] != expected_features:
        raise ValueError(
            f"{dataset_name} entity {path.name} must have {expected_features} "
            f"features, got {values.shape[1]}"
        )
    return values


def _parse_metadata(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"NASA anomaly metadata not found: {path}")
    intervals: Dict[str, List[Tuple[int, int]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"NASA anomaly metadata is empty: {path}")
        normalized = {name.strip().casefold(): name for name in reader.fieldnames}
        chan_column = normalized.get("chan_id")
        sequence_column = normalized.get("anomaly_sequences")
        if chan_column is None or sequence_column is None:
            raise ValueError(
                "NASA metadata requires chan_id and anomaly_sequences columns"
            )
        for row_number, row in enumerate(reader, start=2):
            channel_id = str(row[chan_column]).strip()
            raw_sequences = str(row[sequence_column]).strip()
            if not channel_id:
                raise ValueError(f"NASA metadata row {row_number} has empty chan_id")
            if not raw_sequences or raw_sequences.casefold() in {"nan", "none", "null"}:
                parsed = []
            else:
                try:
                    parsed = ast.literal_eval(raw_sequences)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid anomaly_sequences for {channel_id!r}"
                    ) from exc
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(
                    f"anomaly_sequences for {channel_id!r} must be a list"
                )
            channel_intervals = intervals.setdefault(channel_id, [])
            for interval in parsed:
                if (
                    not isinstance(interval, (list, tuple))
                    or len(interval) != 2
                    or not all(
                        isinstance(value, (int, np.integer)) for value in interval
                    )
                ):
                    raise ValueError(
                        f"Invalid anomaly interval for {channel_id!r}: {interval!r}"
                    )
                channel_intervals.append((int(interval[0]), int(interval[1])))
    return intervals


def load_nasa_data(
    root_path: str,
    *,
    dataset_name: str,
    expected_features: int,
) -> LoadedTimeSeries:
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"{dataset_name} root directory not found: {root}")
    train_dir = root / "train"
    test_dir = root / "test"
    train_files = {path.stem: path for path in sorted(train_dir.glob("*.npy"))}
    test_files = {path.stem: path for path in sorted(test_dir.glob("*.npy"))}
    if not train_files:
        raise FileNotFoundError(f"No {dataset_name} training .npy files in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No {dataset_name} test .npy files in {test_dir}")
    if set(train_files) != set(test_files):
        missing_train = sorted(set(test_files) - set(train_files))
        missing_test = sorted(set(train_files) - set(test_files))
        raise ValueError(
            f"{dataset_name} train/test entity mismatch: "
            f"missing_train={missing_train}, missing_test={missing_test}"
        )

    metadata = _parse_metadata(root / "labeled_anomalies.csv")
    metadata_ids = set(metadata)
    entity_ids = set(test_files)
    if metadata_ids != entity_ids:
        missing_metadata = sorted(entity_ids - metadata_ids)
        extra_metadata = sorted(metadata_ids - entity_ids)
        raise ValueError(
            f"{dataset_name} metadata chan_id/entity mismatch: "
            f"missing metadata={missing_metadata}, "
            f"extra metadata={extra_metadata}"
        )
    train_segments = []
    test_segments = []
    for channel_id in sorted(train_files):
        train_values = _load_entity(
            train_files[channel_id], expected_features, dataset_name
        )
        test_values = _load_entity(
            test_files[channel_id], expected_features, dataset_name
        )
        labels = np.zeros(test_values.shape[0], dtype=np.int32)
        for start, end in metadata.get(channel_id, []):
            if start < 0 or end < start or end >= test_values.shape[0]:
                raise ValueError(
                    f"{dataset_name} anomaly interval [{start}, {end}] for "
                    f"{channel_id!r} is outside length {test_values.shape[0]}"
                )
            labels[start : end + 1] = 1
        train_segments.append(TimeSeriesSegment(channel_id, train_values))
        test_segments.append(TimeSeriesSegment(channel_id, test_values, labels))
    return LoadedTimeSeries(train_segments, test_segments)
