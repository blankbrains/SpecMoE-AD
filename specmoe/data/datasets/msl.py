"""MSL dataset loader preserving each channel as an independent segment."""

from specmoe.data.datasets._nasa import load_nasa_data
from specmoe.data.types import LoadedTimeSeries


def load_data(root_path: str) -> LoadedTimeSeries:
    return load_nasa_data(
        root_path,
        dataset_name="MSL",
        expected_features=55,
    )
