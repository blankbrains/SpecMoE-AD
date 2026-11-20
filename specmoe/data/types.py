"""Validated segment-aware time-series data contracts."""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


def _validated_values(values: np.ndarray, *, segment_id: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(
            f"Segment {segment_id!r} values must have shape [time, variables], "
            f"got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"Segment {segment_id!r} values cannot be empty")
    try:
        array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Segment {segment_id!r} values must be numeric"
        ) from exc
    if not np.isfinite(array).all():
        invalid_count = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"Segment {segment_id!r} values contain {invalid_count} "
            "non-finite NaN/Inf entries"
        )
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validated_binary_labels(
    labels: np.ndarray,
    *,
    segment_id: str,
    expected_length: int,
) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(
            f"Segment {segment_id!r} labels must be one-dimensional, "
            f"got {array.shape}"
        )
    if array.shape[0] != expected_length:
        raise ValueError(
            f"Segment {segment_id!r} label length {array.shape[0]} does not "
            f"match value length {expected_length}"
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"Segment {segment_id!r} labels must be numeric binary values")
    if not np.isfinite(array).all():
        invalid_count = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"Segment {segment_id!r} labels contain {invalid_count} "
            "non-finite NaN/Inf entries"
        )
    if not np.isin(array, (0, 1)).all():
        unique = np.unique(array)
        raise ValueError(
            f"Segment {segment_id!r} labels must be binary 0/1 before "
            f"integer conversion, got {unique.tolist()}"
        )
    result = np.array(array, dtype=np.int32, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TimeSeriesSegment:
    """One independent contiguous time series."""

    segment_id: str
    values: np.ndarray
    labels: Optional[np.ndarray] = None
    metadata: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError("segment_id must be a non-empty string")
        segment_id = self.segment_id.strip()
        values = _validated_values(self.values, segment_id=segment_id)
        labels = None
        if self.labels is not None:
            labels = _validated_binary_labels(
                self.labels,
                segment_id=segment_id,
                expected_length=values.shape[0],
            )
        if self.metadata is None:
            metadata = MappingProxyType({})
        elif not isinstance(self.metadata, Mapping):
            raise ValueError("segment metadata must be a mapping")
        else:
            metadata = MappingProxyType(deepcopy(dict(self.metadata)))
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", metadata)

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class LoadedTimeSeries:
    """Named independent train and test segments for one benchmark."""

    train_segments: Sequence[TimeSeriesSegment]
    test_segments: Sequence[TimeSeriesSegment]
    metadata: Optional[Mapping[str, object]] = None
    excluded_segments: Sequence[Mapping[str, object]] = ()

    def __post_init__(self) -> None:
        train_segments = self._validated_split("train", self.train_segments)
        test_segments = self._validated_split("test", self.test_segments)
        if self.metadata is None:
            metadata = MappingProxyType({})
        elif not isinstance(self.metadata, Mapping):
            raise ValueError("LoadedTimeSeries metadata must be a mapping")
        else:
            metadata = MappingProxyType(deepcopy(dict(self.metadata)))
        try:
            excluded_records = tuple(self.excluded_segments)
        except TypeError as exc:
            raise ValueError("excluded_segments must be a sequence") from exc
        if not all(isinstance(record, Mapping) for record in excluded_records):
            raise ValueError("excluded_segments must contain only mappings")
        excluded_segments = tuple(
            MappingProxyType(deepcopy(dict(record)))
            for record in excluded_records
        )
        all_segments = train_segments + test_segments
        if not all_segments and not excluded_segments:
            raise ValueError(
                "LoadedTimeSeries must contain an active or excluded segment"
            )
        if all_segments:
            feature_dims = {segment.feature_dim for segment in all_segments}
            if len(feature_dims) != 1:
                raise ValueError(
                    "All train/test segments must have the same feature width; "
                    f"got {sorted(feature_dims)}"
                )
        object.__setattr__(self, "train_segments", train_segments)
        object.__setattr__(self, "test_segments", test_segments)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "excluded_segments", excluded_segments)

    @staticmethod
    def _validated_split(
        split: str,
        segments: Sequence[TimeSeriesSegment],
    ) -> Tuple[TimeSeriesSegment, ...]:
        try:
            result = tuple(segments)
        except TypeError as exc:
            raise ValueError(f"{split}_segments must be a sequence") from exc
        if not all(isinstance(segment, TimeSeriesSegment) for segment in result):
            raise ValueError(
                f"{split}_segments must contain only TimeSeriesSegment instances"
            )
        ids = [segment.segment_id for segment in result]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split}_segments contain duplicate segment IDs")
        return result

    @property
    def feature_dim(self) -> int:
        segments = self.train_segments or self.test_segments
        if not segments:
            raise ValueError("Excluded-only LoadedTimeSeries has no feature dimension")
        return segments[0].feature_dim
