"""Segment-aware sliding-window dataset for SpecMoE-AD."""

from bisect import bisect_right
from numbers import Integral
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from specmoe.data.types import TimeSeriesSegment


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


class TimeSeriesDataset(Dataset):
    """Create windows independently with O(number_of_segments) index state."""

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[TimeSeriesSegment]],
        window_size: int,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
        scaler: Optional[object] = None,
        fit_scaler: bool = False,
        cover_end: bool = False,
    ):
        self.window_size = _positive_integer(window_size, name="window_size")
        self.stride = _positive_integer(stride, name="stride")
        if not isinstance(fit_scaler, (bool, np.bool_)):
            raise ValueError("fit_scaler must be boolean")
        if fit_scaler and scaler is None:
            raise ValueError("fit_scaler=True requires a scaler")
        if not isinstance(cover_end, (bool, np.bool_)):
            raise ValueError("cover_end must be boolean")
        self.cover_end = bool(cover_end)

        segments = self._coerce_segments(data, labels)
        if scaler is not None:
            if fit_scaler:
                scaler.fit_segments(segments)
            segments = list(scaler.transform(segments))

        self.scaler = scaler
        self.segments = tuple(segments)
        self.data = tuple(segment.values for segment in self.segments)
        self.num_samples_orig = len(self.segments)
        self.num_vars = self.segments[0].feature_dim

        labels_present = [segment.labels is not None for segment in self.segments]
        if any(labels_present) and not all(labels_present):
            raise ValueError(
                "Either every segment must have labels or none may have labels"
            )
        self.has_labels = all(labels_present)

        regular_counts = []
        window_counts = []
        cumulative_offsets = [0]
        for segment in self.segments:
            if segment.length < self.window_size:
                regular_count = 0
                total_count = 0
            else:
                regular_count = (
                    segment.length - self.window_size
                ) // self.stride + 1
                final_regular_start = (regular_count - 1) * self.stride
                final_start = segment.length - self.window_size
                add_end_window = (
                    self.cover_end and final_regular_start != final_start
                )
                total_count = regular_count + int(add_end_window)
            regular_counts.append(regular_count)
            window_counts.append(total_count)
            cumulative_offsets.append(cumulative_offsets[-1] + total_count)

        self._segment_regular_window_counts = tuple(regular_counts)
        self._segment_window_counts = tuple(window_counts)
        self._cumulative_window_offsets = tuple(cumulative_offsets)
        self._total_windows = cumulative_offsets[-1]
        if self._total_windows == 0:
            raise ValueError(
                "No eligible windows: every segment is shorter than window_size"
            )

    @staticmethod
    def _coerce_segments(data, labels) -> list:
        if isinstance(data, np.ndarray):
            array = np.asarray(data)
            if array.ndim == 2:
                label_array = None if labels is None else np.asarray(labels)
                return [TimeSeriesSegment("0", array, label_array)]
            if array.ndim == 3:
                if array.shape[0] == 0:
                    raise ValueError("3D data must contain at least one sample")
                label_array = None if labels is None else np.asarray(labels)
                if label_array is not None:
                    if (
                        label_array.ndim != 2
                        or label_array.shape[:2] != array.shape[:2]
                    ):
                        raise ValueError(
                            "Labels for 3D data must have shape [samples, time]"
                        )
                return [
                    TimeSeriesSegment(
                        str(index),
                        array[index],
                        None if label_array is None else label_array[index],
                    )
                    for index in range(array.shape[0])
                ]
            raise ValueError(
                f"Expected data to be 2D or 3D, got shape {array.shape}"
            )
        if labels is not None:
            raise ValueError("labels must be stored on TimeSeriesSegment inputs")
        try:
            segments = list(data)
        except TypeError as exc:
            raise ValueError(
                "data must be an ndarray or a sequence of TimeSeriesSegment"
            ) from exc
        if not segments:
            raise ValueError("At least one time-series segment is required")
        if not all(isinstance(segment, TimeSeriesSegment) for segment in segments):
            raise ValueError("Segment input contains a non-TimeSeriesSegment value")
        feature_dims = {segment.feature_dim for segment in segments}
        if len(feature_dims) != 1:
            raise ValueError(
                "All segments must have the same feature width, "
                f"got {sorted(feature_dims)}"
            )
        return segments

    def _resolve_window(self, idx: int) -> Tuple[int, int]:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral):
            raise IndexError(f"Window index must be an integer, got {idx!r}")
        idx = int(idx)
        if idx < 0 or idx >= self._total_windows:
            raise IndexError(
                f"Index {idx} out of range [0, {self._total_windows})"
            )
        segment_index = bisect_right(
            self._cumulative_window_offsets,
            idx,
        ) - 1
        local_index = idx - self._cumulative_window_offsets[segment_index]
        regular_count = self._segment_regular_window_counts[segment_index]
        if local_index < regular_count:
            start_t = local_index * self.stride
        else:
            start_t = (
                self.segments[segment_index].length - self.window_size
            )
        return segment_index, start_t

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        segment_index, start_t = self._resolve_window(idx)
        segment = self.segments[segment_index]
        x = segment.values[start_t : start_t + self.window_size]
        result = (torch.from_numpy(x.copy()),)
        if not self.has_labels:
            return result
        window_labels = segment.labels[start_t : start_t + self.window_size]
        label = torch.tensor(
            float(window_labels.max() > 0),
            dtype=torch.float32,
        )
        return result + (label,)

    def get_full_series(self, sample_idx: int = 0) -> np.ndarray:
        return self.segments[sample_idx].values

    def get_window_location(self, idx: int) -> Tuple[str, int]:
        segment_index, start_t = self._resolve_window(idx)
        return self.segments[segment_index].segment_id, start_t

    @property
    def n_windows(self) -> int:
        return self._total_windows

    @property
    def feature_dim(self) -> int:
        return self.num_vars


class ValidationWindowDataset(Dataset):
    """Window independent validation segments without altering short segments.

    Segments at least ``window_size`` long use regular fixed-size windows.
    A shorter segment is emitted once at its original full length. Callers
    must batch samples only with matching temporal lengths.
    """

    def __init__(
        self,
        segments: Sequence[TimeSeriesSegment],
        *,
        window_size: int,
        stride: int = 1,
        scaler: Optional[object] = None,
    ):
        self.window_size = _positive_integer(window_size, name="window_size")
        self.stride = _positive_integer(stride, name="stride")
        resolved_segments = TimeSeriesDataset._coerce_segments(segments, None)
        if scaler is not None:
            resolved_segments = list(scaler.transform(resolved_segments))
        self.scaler = scaler
        self.segments = tuple(resolved_segments)
        self.data = tuple(segment.values for segment in self.segments)
        self.num_samples_orig = len(self.segments)
        self.num_vars = self.segments[0].feature_dim

        labels_present = [segment.labels is not None for segment in self.segments]
        if any(labels_present) and not all(labels_present):
            raise ValueError(
                "Either every segment must have labels or none may have labels"
            )
        self.has_labels = all(labels_present)

        windows = []
        for segment_index, segment in enumerate(self.segments):
            if segment.length < self.window_size:
                windows.append((segment_index, 0, segment.length))
                continue
            for start in range(
                0,
                segment.length - self.window_size + 1,
                self.stride,
            ):
                windows.append((segment_index, start, self.window_size))
        if not windows:
            raise ValueError("Validation segments did not produce any windows")
        self._windows = tuple(windows)
        self.window_lengths = tuple(length for _, _, length in self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral):
            raise IndexError(f"Window index must be an integer, got {idx!r}")
        idx = int(idx)
        if idx < 0 or idx >= len(self._windows):
            raise IndexError(f"Index {idx} out of range [0, {len(self._windows)})")
        segment_index, start_t, length = self._windows[idx]
        segment = self.segments[segment_index]
        x = segment.values[start_t : start_t + length]
        result = (torch.from_numpy(x.copy()),)
        if not self.has_labels:
            return result
        window_labels = segment.labels[start_t : start_t + length]
        label = torch.tensor(
            float(window_labels.max() > 0),
            dtype=torch.float32,
        )
        return result + (label,)

    def get_window_location(self, idx: int) -> Tuple[str, int]:
        segment_index, start_t, _length = self._windows[idx]
        return self.segments[segment_index].segment_id, start_t

    def get_window_length(self, idx: int) -> int:
        return int(self._windows[idx][2])

    @property
    def n_windows(self) -> int:
        return len(self._windows)

    @property
    def feature_dim(self) -> int:
        return self.num_vars
