"""Strict SMD loader preserving all 28 machine boundaries."""

from pathlib import Path

from specmoe.data.datasets._io import load_label_vector, load_numeric_matrix
from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


_SMD_GROUPS = {1: 8, 2: 9, 3: 11}
_FEATURE_COUNT = 38


def _load_features(path: Path):
    values = load_numeric_matrix(path)
    if values.shape[1] != _FEATURE_COUNT:
        raise ValueError(
            f"SMD file {path.name} must have {_FEATURE_COUNT} features, "
            f"got {values.shape[1]}"
        )
    return values


def load_data(root_path: str) -> LoadedTimeSeries:
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"SMD root directory not found: {root}")
    train_segments = []
    test_segments = []
    for group, count in _SMD_GROUPS.items():
        for index in range(1, count + 1):
            segment_id = f"machine-{group}-{index}"
            train_values = _load_features(root / "train" / f"{segment_id}.txt")
            test_values = _load_features(root / "test" / f"{segment_id}.txt")
            labels = load_label_vector(
                root / "test_label" / f"{segment_id}.txt"
            )
            train_segments.append(TimeSeriesSegment(segment_id, train_values))
            test_segments.append(
                TimeSeriesSegment(segment_id, test_values, labels)
            )
    return LoadedTimeSeries(train_segments, test_segments)
