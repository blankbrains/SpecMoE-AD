"""Segment-aware dataset loader registry.

Every loader returns :class:`specmoe.data.types.LoadedTimeSeries` and
preserves each independent machine, channel, or curve as a separate segment.
"""

from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment
from specmoe.data.datasets.smd import load_data as load_smd
from specmoe.data.datasets.msl import load_data as load_msl
from specmoe.data.datasets.swat import load_data as load_swat
from specmoe.data.datasets.wadi import load_data as load_wadi
from specmoe.data.datasets.psm import load_data as load_psm

DATASET_REGISTRY = {
    "smd": load_smd,
    "msl": load_msl,
    "swat": load_swat,
    "wadi": load_wadi,
    "psm": load_psm,
}

__all__ = [
    "load_smd",
    "load_msl",
    "load_swat",
    "load_wadi",
    "load_psm",
    "DATASET_REGISTRY",
    "LoadedTimeSeries",
    "TimeSeriesSegment",
]
