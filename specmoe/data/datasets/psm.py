"""Strict single-segment PSM dataset loader."""

from pathlib import Path

from specmoe.data.datasets._io import load_label_vector, load_numeric_matrix
from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


_FEATURE_COUNT = 25


def load_data(root_path: str) -> LoadedTimeSeries:
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"PSM root directory not found: {root}")
    train_values = load_numeric_matrix(root / "train.csv")
    test_values = load_numeric_matrix(root / "test.csv")
    labels = load_label_vector(root / "test_label.csv")
    for split, values in (("train", train_values), ("test", test_values)):
        if values.shape[1] != _FEATURE_COUNT:
            raise ValueError(
                f"PSM {split} data must have {_FEATURE_COUNT} feature columns, "
                f"got {values.shape[1]}"
            )
    return LoadedTimeSeries(
        train_segments=[TimeSeriesSegment("psm", train_values)],
        test_segments=[TimeSeriesSegment("psm", test_values, labels)],
    )
