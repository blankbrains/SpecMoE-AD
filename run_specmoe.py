"""Command-line training and detection entry point for SpecMoE-AD."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specmoe.data.dataset import TimeSeriesDataset, ValidationWindowDataset
from specmoe.data.preprocess import StandardScaler
from specmoe.data.protocol import (
    ProtocolSplit,
    split_paper_partitions,
)
from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment
from specmoe.eval.scorer import (
    calibrate_threshold,
    detect_label,
    resolve_detection_threshold,
)
from specmoe.model.loading import (
    compute_source_fingerprint,
    load_model_checkpoint,
)
from specmoe.model.m2b_moe_model import M2BMoEModel
from specmoe.train.trainer import (
    FIXED_VALIDATION_COMPONENT_WEIGHTS,
    FIXED_VALIDATION_OBJECTIVE_REVISION,
    Trainer,
    make_dataloader_generator,
    seed_dataloader_worker,
    set_deterministic_seed,
)

ALLOWED_EXPERIMENT_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "data_root",
        "n_features",
        "seed",
        "validation_seed",
        "window_size",
        "stride",
        "d_model",
        "d_ff",
        "patch_size",
        "patch_stride",
        "bottleneck_dims",
        "cross_var_heads",
        "use_revin",
        "input_mask_ratio",
        "mask_mode",
        "lambda_freq",
        "router_hidden",
        "router_mode",
        "expert_specialization",
        "diversity_weight",
        "diversity_mode",
        "load_balance_weight",
        "grad_weight",
        "score_freq_weight",
        "disagreement_weight",
        "use_grad_score",
        "use_adaptive_patches",
        "use_learnable_temp",
        "stft_window_sizes",
        "stft_hop_ratio",
        "frequency_score_aggregation",
        "lr",
        "batch_size",
        "num_epochs",
        "patience",
        "scheduler_patience",
        "warmup_epochs",
        "warmup_start_factor",
        "gradient_accumulation_steps",
        "loss_warmup_epochs",
        "validation_rounds",
        "calibration_percentile",
        "weight_decay",
        "grad_clip",
        "device",
        "save_dir",
    }
)


def _load_yaml_config(config_path: str) -> dict:
    """Load a YAML experiment configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to use --config"
        ) from exc
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file {config_path} is not a YAML mapping"
        )
    unknown = sorted(set(config) - ALLOWED_EXPERIMENT_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Config contains unknown or unsupported fields: "
            + ", ".join(unknown)
        )
    return config


def _resolve_param(cli_value, config_value, default):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _resolve_training_device(
    cli_device: Optional[str],
    config_device: Optional[str],
) -> tuple[str, str]:
    requested = _resolve_param(cli_device, config_device, "auto")
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be one of: auto, cpu, cuda")
    requested = requested.strip().lower()
    if requested == "auto":
        return (
            requested,
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    try:
        parsed = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"Invalid device request {requested!r}"
        ) from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("device must resolve to CPU or CUDA")
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested but CUDA is unavailable"
        )
    return requested, str(parsed)


def load_csv_data(filepath: str) -> np.ndarray:
    """Load a finite numeric [time, variables] CSV array."""
    import csv

    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            first_row = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {filepath}") from exc
    try:
        [float(value) for value in first_row]
        has_header = False
    except ValueError:
        has_header = True
    data = np.loadtxt(
        filepath,
        delimiter=",",
        skiprows=1 if has_header else 0,
    )
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"CSV data must have non-empty [time, variables] shape, "
            f"got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError("CSV data contains non-finite NaN/Inf values")
    return data.astype(np.float32, copy=False)


def load_npy_data(filepath: str) -> np.ndarray:
    """Load a finite numeric [time, variables] NumPy array."""
    data = np.load(filepath, allow_pickle=False)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"NumPy data must have non-empty [time, variables] shape, "
            f"got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError("NumPy data contains non-finite NaN/Inf values")
    return data.astype(np.float32, copy=False)


def load_dataset_from_config(config: Mapping) -> LoadedTimeSeries:
    """Load one registered benchmark as independent train/test segments."""
    from specmoe.data.datasets import DATASET_REGISTRY

    dataset_name = config.get("dataset")
    data_root = config.get("data_root")
    if not dataset_name:
        raise ValueError(
            "Config must specify dataset; available values are "
            + ", ".join(DATASET_REGISTRY)
        )
    if not data_root:
        raise ValueError("Config must specify data_root")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; "
            f"available values are {list(DATASET_REGISTRY)}"
        )
    loaded = DATASET_REGISTRY[dataset_name](data_root)
    if not isinstance(loaded, LoadedTimeSeries):
        raise TypeError(
            f"Dataset loader {dataset_name!r} must return LoadedTimeSeries"
        )
    return loaded


@dataclass(frozen=True)
class WindowIndexRange:
    segment_id: str
    first_base_index: int
    count: int


class IndexedWindowDataset(Dataset):
    """Select eligible windows and retain original segment coordinates."""

    def __init__(
        self,
        base_dataset: TimeSeriesDataset,
        window_ranges: Sequence[WindowIndexRange],
        origin_offsets: Mapping[str, int],
    ):
        ranges = tuple(window_ranges)
        if not ranges:
            raise ValueError("No eligible normal windows are available")
        if not all(
            isinstance(item, WindowIndexRange)
            and item.first_base_index >= 0
            and item.count > 0
            for item in ranges
        ):
            raise ValueError("window_ranges contains an invalid range")
        self.base_dataset = base_dataset
        self.window_ranges = ranges
        self.origin_offsets = dict(origin_offsets)
        self.window_size = base_dataset.window_size
        self.num_vars = base_dataset.num_vars
        self.scaler = base_dataset.scaler
        cumulative = [0]
        for item in ranges:
            cumulative.append(cumulative[-1] + item.count)
        self._cumulative_counts = tuple(cumulative)

    def __len__(self) -> int:
        return self._cumulative_counts[-1]

    def _base_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise IndexError("Window index must be an integer")
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError("Window index is out of range")
        range_index = bisect_right(self._cumulative_counts, index) - 1
        item = self.window_ranges[range_index]
        offset = index - self._cumulative_counts[range_index]
        return item.first_base_index + offset

    def __getitem__(self, index: int):
        return self.base_dataset[self._base_index(index)]

    def get_window_location(self, index: int):
        segment_id, local_start = self.base_dataset.get_window_location(
            self._base_index(index)
        )
        return (
            segment_id,
            local_start + self.origin_offsets[segment_id],
        )

    def single_segment_views(self):
        segment_by_id = {
            segment.segment_id: segment
            for segment in self.base_dataset.segments
        }
        views = []
        logical_start = 0
        range_start = 0
        while range_start < len(self.window_ranges):
            segment_id = self.window_ranges[range_start].segment_id
            range_stop = range_start
            count = 0
            while (
                range_stop < len(self.window_ranges)
                and self.window_ranges[range_stop].segment_id == segment_id
            ):
                count += self.window_ranges[range_stop].count
                range_stop += 1
            views.append(
                SingleSegmentWindowView(
                    self,
                    parent_start=logical_start,
                    count=count,
                    segment_id=segment_id,
                    origin=self.origin_offsets[segment_id],
                    segment_length=segment_by_id[segment_id].length,
                    window_ranges=self.window_ranges[
                        range_start:range_stop
                    ],
                )
            )
            logical_start += count
            range_start = range_stop
        return views


class SingleSegmentWindowView(Dataset):
    """A segment-local view over a possibly multi-segment window dataset."""

    def __init__(
        self,
        parent: Dataset,
        *,
        parent_start: int,
        count: int,
        segment_id: str,
        origin: int,
        segment_length: int,
        window_ranges: Sequence[WindowIndexRange],
    ):
        self.parent = parent
        self.parent_start = int(parent_start)
        self.count = int(count)
        if self.count < 1:
            raise ValueError("Single-segment view cannot be empty")
        self.window_ranges = tuple(window_ranges)
        self.segment_id = segment_id
        self.origin = int(origin)
        self.segment_length = int(segment_length)
        self.window_size = int(parent.window_size)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        if index < 0 or index >= self.count:
            raise IndexError("Window index is out of range")
        return self.parent[self.parent_start + index]

    def get_window_location(self, index: int):
        segment_id, start = self.parent.get_window_location(
            self.parent_start + index
        )
        if segment_id != self.segment_id:
            raise RuntimeError("Single-segment view contains another segment")
        return segment_id, start - self.origin


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Create deterministic batches whose windows share one temporal length."""

    def __init__(self, dataset: Dataset, *, batch_size: int):
        self.batch_size = _validated_positive_integer(
            batch_size,
            name="batch_size",
        )
        get_window_length = getattr(dataset, "get_window_length", None)
        if not callable(get_window_length):
            raise TypeError(
                "LengthBucketBatchSampler requires get_window_length(index)"
            )
        buckets: Dict[int, list[int]] = {}
        for index in range(len(dataset)):
            length = _validated_positive_integer(
                get_window_length(index),
                name="validation window length",
            )
            buckets.setdefault(length, []).append(index)
        self._batches = tuple(
            batch
            for length in sorted(buckets)
            for start in range(0, len(buckets[length]), self.batch_size)
            for batch in [buckets[length][start : start + self.batch_size]]
        )
        if not self._batches:
            raise ValueError("Validation dataset cannot be empty")

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


@dataclass(frozen=True)
class PreparedTrainingData:
    train_dataset: IndexedWindowDataset
    val_dataset: Optional[Dataset]
    scaler: StandardScaler


def _validated_positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _validated_nonnegative_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return resolved


def _validated_unit_interval_factor(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and in (0, 1]")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and in (0, 1]"
        ) from exc
    if not np.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return resolved


def _eligible_normal_window_ranges(
    dataset: TimeSeriesDataset,
) -> Sequence[WindowIndexRange]:
    segment_by_id = {
        segment.segment_id: segment for segment in dataset.segments
    }
    ranges = []
    active_segment = None
    active_start = None
    active_count = 0
    previous_index = None
    for index in range(len(dataset)):
        segment_id, start = dataset.get_window_location(index)
        labels = segment_by_id[segment_id].labels
        eligible = labels is None or not bool(
            labels[start : start + dataset.window_size].any()
        )
        if eligible:
            contiguous = (
                active_segment == segment_id
                and previous_index is not None
                and index == previous_index + 1
            )
            if not contiguous:
                if active_count:
                    ranges.append(
                        WindowIndexRange(
                            active_segment,
                            active_start,
                            active_count,
                        )
                    )
                active_segment = segment_id
                active_start = index
                active_count = 0
            active_count += 1
            previous_index = index
        elif active_count:
            ranges.append(
                WindowIndexRange(
                    active_segment,
                    active_start,
                    active_count,
                )
            )
            active_segment = None
            active_start = None
            active_count = 0
            previous_index = None
    if active_count:
        ranges.append(
            WindowIndexRange(
                active_segment,
                active_start,
                active_count,
            )
        )
    return ranges


def _scaler_segments_for_normal_windows(
    dataset: TimeSeriesDataset,
    window_ranges: Sequence[WindowIndexRange],
) -> Sequence[TimeSeriesSegment]:
    """Materialize only points covered by at least one normal train window."""
    if dataset.cover_end:
        raise ValueError(
            "Normal-window scaler construction requires regular train windows"
        )
    intervals_by_segment: Dict[str, list[tuple[int, int]]] = {
        segment.segment_id: [] for segment in dataset.segments
    }
    for window_range in window_ranges:
        first_segment_id, first_start = dataset.get_window_location(
            window_range.first_base_index
        )
        last_segment_id, last_start = dataset.get_window_location(
            window_range.first_base_index + window_range.count - 1
        )
        if (
            first_segment_id != window_range.segment_id
            or last_segment_id != window_range.segment_id
        ):
            raise RuntimeError(
                "Normal window range crosses an unexpected segment boundary"
            )
        if dataset.stride <= dataset.window_size:
            intervals_by_segment[first_segment_id].append(
                (first_start, last_start + dataset.window_size)
            )
        else:
            for start in range(
                first_start,
                last_start + dataset.stride,
                dataset.stride,
            ):
                intervals_by_segment[first_segment_id].append(
                    (start, start + dataset.window_size)
                )

    segment_by_id = {
        segment.segment_id: segment for segment in dataset.segments
    }
    scaler_segments = []
    for segment_id, intervals in intervals_by_segment.items():
        if not intervals:
            continue
        merged = []
        for start, stop in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, stop])
            else:
                merged[-1][1] = max(merged[-1][1], stop)
        source = segment_by_id[segment_id]
        for index, (start, stop) in enumerate(merged):
            scaler_segments.append(
                TimeSeriesSegment(
                    f"{segment_id}:scaler:{index}",
                    source.values[start:stop],
                )
            )
    if not scaler_segments:
        raise ValueError(
            "No points are covered by eligible normal training windows"
        )
    return tuple(scaler_segments)


def prepare_training_data(
    protocol_split: ProtocolSplit,
    *,
    window_size: int,
    stride: int,
    seed: int,
) -> PreparedTrainingData:
    """Build protocol-defined train and validation datasets."""
    if not isinstance(protocol_split, ProtocolSplit):
        raise TypeError("protocol_split must be a ProtocolSplit")
    window_size = _validated_positive_integer(
        window_size, name="window_size"
    )
    stride = _validated_positive_integer(stride, name="stride")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, Integral)
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    unscaled_train_base = TimeSeriesDataset(
        protocol_split.train_segments,
        window_size=window_size,
        stride=stride,
    )
    train_ranges = _eligible_normal_window_ranges(unscaled_train_base)
    scaler = StandardScaler().fit_segments(
        _scaler_segments_for_normal_windows(
            unscaled_train_base,
            train_ranges,
        )
    )
    train_base = TimeSeriesDataset(
        protocol_split.train_segments,
        window_size=window_size,
        stride=stride,
        scaler=scaler,
        fit_scaler=False,
    )
    train_offsets = {
        segment.segment_id: 0 for segment in protocol_split.train_segments
    }
    train_dataset = IndexedWindowDataset(
        train_base,
        train_ranges,
        train_offsets,
    )

    validation_dataset: Optional[Dataset] = ValidationWindowDataset(
        protocol_split.validation_segments,
        window_size=window_size,
        stride=stride,
        scaler=scaler,
    )
    return PreparedTrainingData(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        scaler=scaler,
    )


@torch.no_grad()
def score_single_segment(
    model,
    dataset: Dataset,
    *,
    batch_size: int,
    device: str,
    n_rounds: int = 1,
    require_full_coverage: bool = False,
) -> np.ndarray:
    """Score one segment and overlap-average windows onto its time axis."""
    batch_size = _validated_positive_integer(
        batch_size,
        name="batch_size",
    )
    n_rounds = _validated_positive_integer(
        n_rounds,
        name="n_rounds",
    )
    if isinstance(dataset, SingleSegmentWindowView):
        segment_length = dataset.segment_length
    elif isinstance(dataset, TimeSeriesDataset):
        if len(dataset.segments) != 1:
            raise ValueError(
                "score_single_segment requires exactly one segment"
            )
        segment_length = dataset.segments[0].length
    else:
        raise TypeError(
            "score_single_segment requires TimeSeriesDataset or "
            "SingleSegmentWindowView"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    was_training = bool(getattr(model, "training", False))
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    score_sum = np.zeros(segment_length, dtype=np.float64)
    score_count = np.zeros(segment_length, dtype=np.float64)
    try:
        for _round_index in range(n_rounds):
            window_index = 0
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                result = model.detect(x.to(device))
                scores = result.get("anom_score")
                if not isinstance(scores, torch.Tensor):
                    raise TypeError(
                        "model.detect() must return tensor anom_score"
                    )
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError(
                        "Segment scoring produced non-finite scores"
                    )
                batch_scores = scores.detach().cpu().numpy()
                if (
                    batch_scores.ndim != 2
                    or batch_scores.shape[1] != dataset.window_size
                ):
                    raise ValueError(
                        "model.detect() anom_score must have shape "
                        "[batch, window_size]"
                    )
                for local_index in range(batch_scores.shape[0]):
                    _segment_id, start = dataset.get_window_location(
                        window_index
                    )
                    stop = start + dataset.window_size
                    if start < 0 or stop > segment_length:
                        raise RuntimeError(
                            "Window location lies outside its segment"
                        )
                    score_sum[start:stop] += batch_scores[local_index]
                    score_count[start:stop] += 1.0
                    window_index += 1
    finally:
        train_method = getattr(model, "train", None)
        if callable(train_method):
            train_method(was_training)
    covered = score_count > 0.0
    if not covered.any():
        raise ValueError("Segment scoring produced no covered time points")
    if require_full_coverage and not covered.all():
        first_uncovered = int(np.flatnonzero(~covered)[0])
        raise RuntimeError(
            "Segment scoring did not cover all time points; first "
            f"uncovered index is {first_uncovered}"
        )
    return (score_sum[covered] / score_count[covered]).astype(
        np.float32,
        copy=False,
    )


def _single_segment_views(dataset: Dataset):
    if isinstance(dataset, IndexedWindowDataset):
        return dataset.single_segment_views()
    if isinstance(dataset, TimeSeriesDataset):
        if len(dataset.segments) == 1:
            return [dataset]
        segment_by_id = {
            segment.segment_id: segment for segment in dataset.segments
        }
        views = []
        start_index = 0
        while start_index < len(dataset):
            segment_id, _ = dataset.get_window_location(start_index)
            stop_index = start_index + 1
            while (
                stop_index < len(dataset)
                and dataset.get_window_location(stop_index)[0]
                == segment_id
            ):
                stop_index += 1
            count = stop_index - start_index
            if count:
                views.append(
                    SingleSegmentWindowView(
                        dataset,
                        parent_start=start_index,
                        count=count,
                        segment_id=segment_id,
                        origin=0,
                        segment_length=segment_by_id[segment_id].length,
                        window_ranges=(
                            WindowIndexRange(
                                segment_id,
                                start_index,
                                count,
                            ),
                        ),
                    )
                )
            start_index = stop_index
        return views
    raise TypeError("Unsupported calibration dataset type")


@torch.no_grad()
def collect_calibration_scores(
    model,
    loaders: Sequence[DataLoader],
    *,
    device: str,
) -> np.ndarray:
    """Collect overlap-aggregated normal point scores per real segment."""
    collected = []
    for loader in loaders:
        if loader is None:
            continue
        batch_size = loader.batch_size or 1
        for segment_dataset in _single_segment_views(loader.dataset):
            collected.append(
                score_single_segment(
                    model,
                    segment_dataset,
                    batch_size=batch_size,
                    device=device,
                    n_rounds=1,
                    require_full_coverage=False,
                )
            )
    if not collected:
        raise ValueError("Calibration loaders produced no scores")
    return np.concatenate(collected).astype(np.float64, copy=False)


def _git_revision() -> str:
    repository = os.path.dirname(os.path.abspath(__file__))
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        explicit_revision = os.environ.get("SPECMOE_GIT_REVISION", "").strip()
        if explicit_revision:
            return explicit_revision
        try:
            source_fingerprint = compute_source_fingerprint(repository)
        except (OSError, ValueError) as fingerprint_exc:
            raise RuntimeError(
                "Unable to resolve Git revision or source fingerprint"
            ) from fingerprint_exc
        return f"source-fingerprint:{source_fingerprint}"
    if not revision:
        raise RuntimeError("Git returned an empty revision")
    return revision + ("-dirty" if dirty else "")


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _write_run_config(path: str, resolved_config: Mapping) -> None:
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".run-config-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = file.name
            json.dump(
                _json_safe(resolved_config),
                file,
                indent=2,
                sort_keys=True,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_custom_training_data(args) -> LoadedTimeSeries:
    data_path = args.data
    if data_path.endswith(".csv"):
        data = load_csv_data(data_path)
    elif data_path.endswith(".npy"):
        data = load_npy_data(data_path)
    else:
        raise ValueError(
            f"Unsupported file format: {data_path}. Use .csv or .npy"
        )
    labels = None
    labels_path = getattr(args, "labels", None)
    if labels_path:
        if labels_path.endswith(".csv"):
            labels = load_csv_data(labels_path)
        else:
            labels = np.load(labels_path, allow_pickle=False)
        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        elif labels.ndim != 1:
            raise ValueError(
                "Custom labels must have shape [time] or [time, 1]"
            )
    return LoadedTimeSeries(
        train_segments=[TimeSeriesSegment("custom", data, labels)],
        test_segments=[],
    )


def _build_training_protocol(
    loaded: LoadedTimeSeries,
    *,
    window_size: int,
) -> ProtocolSplit:
    """Build the paper 50/10/40 chronological source partitions."""
    return split_paper_partitions(loaded, window_size=window_size)


def train(args) -> None:
    """Run segmented, deterministic M2B-MoE training."""
    print("=" * 70)
    print("  SpecMoE-AD Training")
    print("=" * 70)

    config = {}
    config_path = getattr(args, "config", None)
    data_path = getattr(args, "data", None)
    if config_path and data_path:
        raise ValueError("--config and --data are mutually exclusive")
    if config_path:
        config = _load_yaml_config(config_path)
        print(f"Config loaded: {config_path}")

    def cfg(key, default=None):
        return _resolve_param(
            getattr(args, key, None),
            config.get(key),
            default,
        )

    window_size = _validated_positive_integer(
        cfg("window_size", 256),
        name="window_size",
    )
    stride = _validated_positive_integer(cfg("stride", 64), name="stride")
    batch_size = _validated_positive_integer(
        cfg("batch_size", 16),
        name="batch_size",
    )
    num_epochs = _validated_positive_integer(
        cfg("num_epochs", 50),
        name="num_epochs",
    )
    patience = _validated_positive_integer(
        cfg("patience", 10),
        name="patience",
    )
    if patience >= num_epochs:
        raise ValueError("patience must be smaller than num_epochs")
    scheduler_patience = _validated_positive_integer(
        cfg("scheduler_patience", max(1, patience // 2)),
        name="scheduler_patience",
    )
    if scheduler_patience >= patience:
        raise ValueError(
            "scheduler_patience must be smaller than patience"
        )
    warmup_epochs = _validated_nonnegative_integer(
        cfg("warmup_epochs", 0),
        name="warmup_epochs",
    )
    warmup_start_factor = _validated_unit_interval_factor(
        cfg("warmup_start_factor", 0.1),
        name="warmup_start_factor",
    )
    gradient_accumulation_steps = _validated_positive_integer(
        cfg("gradient_accumulation_steps", 1),
        name="gradient_accumulation_steps",
    )
    loss_warmup_epochs = _validated_nonnegative_integer(
        cfg("loss_warmup_epochs", 0),
        name="loss_warmup_epochs",
    )
    if warmup_epochs > num_epochs:
        raise ValueError("warmup_epochs cannot exceed num_epochs")
    if loss_warmup_epochs > num_epochs:
        raise ValueError("loss_warmup_epochs cannot exceed num_epochs")
    seed = int(cfg("seed", 42))
    validation_seed = int(cfg("validation_seed", seed + 1_000_003))
    validation_rounds = _validated_positive_integer(
        cfg("validation_rounds", 3),
        name="validation_rounds",
    )
    calibration_percentile = float(
        cfg("calibration_percentile", 99.0)
    )
    if not 0.0 < calibration_percentile < 100.0:
        raise ValueError(
            "calibration_percentile must be in (0, 100)"
        )
    set_deterministic_seed(seed)

    if config.get("dataset") and config.get("data_root"):
        loaded = load_dataset_from_config(config)
        dataset_name = str(config["dataset"])
        data_root = str(config["data_root"])
    elif data_path:
        loaded = _load_custom_training_data(args)
        dataset_name = "custom"
        data_root = os.path.abspath(args.data)
    else:
        raise ValueError(
            "No data source specified. Use --config or --data"
        )
    configured_features = config.get("n_features")
    if (
        configured_features is not None
        and int(configured_features) != loaded.feature_dim
    ):
        raise ValueError(
            f"Config declares n_features={configured_features}, but "
            f"loaded data has {loaded.feature_dim} variables"
        )

    protocol_split = _build_training_protocol(
        loaded,
        window_size=window_size,
    )
    prepared = prepare_training_data(
        protocol_split,
        window_size=window_size,
        stride=stride,
        seed=seed,
    )
    train_generator = make_dataloader_generator(seed)
    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        worker_init_fn=seed_dataloader_worker,
    )
    val_loader = (
        DataLoader(
            prepared.val_dataset,
            batch_sampler=LengthBucketBatchSampler(
                prepared.val_dataset,
                batch_size=batch_size,
            ),
            worker_init_fn=seed_dataloader_worker,
        )
        if prepared.val_dataset is not None
        else None
    )
    protocol_manifest = protocol_split.manifest
    print(
        "50/10/40 source partitions="
        f"{len(protocol_manifest['sources'])}, "
        f"normal train windows={len(prepared.train_dataset)}, "
        "validation windows="
        f"{0 if prepared.val_dataset is None else len(prepared.val_dataset)}"
    )

    bottleneck_dims = cfg("bottleneck_dims", [2, 8, 32])
    if isinstance(bottleneck_dims, str):
        bottleneck_dims = json.loads(bottleneck_dims)
    model = M2BMoEModel(
        time_len=window_size,
        num_vars=loaded.feature_dim,
        patch_len=cfg("patch_size", 8),
        patch_stride=cfg("patch_stride", 4),
        d_model=cfg("d_model", 32),
        enc_hidden=cfg("d_ff", 64),
        bottleneck_dims=bottleneck_dims,
        cross_var_heads=cfg("cross_var_heads", 4),
        use_revin=cfg("use_revin", False),
        input_mask_ratio=cfg("input_mask_ratio", 0.5),
        mask_mode=cfg("mask_mode", "contiguous"),
        lambda_freq=cfg("lambda_freq", 1.0),
        router_hidden=cfg("router_hidden", 32),
        router_mode=cfg("router_mode", "variable"),
        expert_specialization=cfg("expert_specialization", "frequency_bands"),
        diversity_weight=cfg("diversity_weight", 0.1),
        load_balance_weight=cfg("load_balance_weight", 0.01),
        grad_weight=cfg("grad_weight", 0.3),
        score_freq_weight=cfg("score_freq_weight", 0.3),
        disagreement_weight=cfg("disagreement_weight", 0.1),
        diversity_mode=cfg("diversity_mode", "contrastive"),
        use_grad_score=cfg("use_grad_score", False),
        use_adaptive_patches=cfg("use_adaptive_patches", True),
        use_learnable_temp=cfg("use_learnable_temp", True),
        stft_window_sizes=cfg("stft_window_sizes", None),
        stft_hop_ratio=cfg("stft_hop_ratio", 0.25),
        frequency_score_aggregation=cfg(
            "frequency_score_aggregation",
            "max",
        ),
    )
    model_config = model.get_config()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Model: M2B-MoE v3, parameters={parameter_count:,}")

    lr = float(cfg("lr", 1e-3))
    weight_decay = float(cfg("weight_decay", 1e-5))
    grad_clip = float(cfg("grad_clip", 5.0))
    requested_device, device = _resolve_training_device(
        getattr(args, "device", None),
        config.get("device"),
    )
    save_dir = (
        cfg("save_dir", None)
        or getattr(args, "save_dir", None)
        or "checkpoints"
    )
    os.makedirs(save_dir, exist_ok=True)

    reproducibility = {
        "seed": seed,
        "validation_seed": validation_seed,
        "validation_rounds": validation_rounds,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    git_revision = _git_revision()
    source_fingerprint = compute_source_fingerprint()
    resolved_config = {
        "schema_version": 1,
        "seed": seed,
        "git_revision": git_revision,
        "source_fingerprint": source_fingerprint,
        "device": {
            "requested": requested_device,
            "resolved": device,
        },
        "dataset": {
            "name": dataset_name,
            "data_root": data_root,
            "n_features": loaded.feature_dim,
            "source_train_segment_count": len(loaded.train_segments),
            "source_test_segment_count": len(loaded.test_segments),
            "excluded_segments": [
                dict(record) for record in loaded.excluded_segments
            ],
        },
        "data_protocol": {
            "protocol_version": protocol_manifest["protocol_version"],
            "manifest": _json_safe(protocol_split.manifest),
            "normal_windows_only": True,
            "window_size": window_size,
            "stride": stride,
            "short_segment_window_policy": (
                "full_segment_when_shorter"
            ),
            "train_windows": len(prepared.train_dataset),
            "val_windows": (
                0
                if prepared.val_dataset is None
                else len(prepared.val_dataset)
            ),
            "test_segment_count": len(protocol_split.test_segments),
            "scaler": {
                "type": "StandardScaler",
                "fit_scope": "normal_training_window_covered_points_only",
                "eps": float(prepared.scaler.eps),
                "n_features": loaded.feature_dim,
            },
        },
        "model": model_config,
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
        },
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": scheduler_patience,
            "min_lr": 1e-6,
        },
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "early_stopping_patience": patience,
            "scheduler_patience": scheduler_patience,
            "grad_clip": grad_clip,
            "warmup_epochs": warmup_epochs,
            "warmup_start_factor": warmup_start_factor,
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "loss_warmup_epochs": loss_warmup_epochs,
            "train_compute_scores": False,
        },
        "validation": {
            "seed": validation_seed,
            "rounds": validation_rounds,
            "masked": True,
            "source": "source_chronological_validation_10_percent",
            "objective_revision": (
                FIXED_VALIDATION_OBJECTIVE_REVISION
            ),
            "component_weights": dict(
                FIXED_VALIDATION_COMPONENT_WEIGHTS
            ),
        },
        "calibration": {
            "source": "normal_training_candidate_windows_only",
            "percentile": calibration_percentile,
            "threshold": None,
        },
    }
    run_config_path = os.path.join(save_dir, "run_config.json")
    _write_run_config(run_config_path, resolved_config)

    trainer = Trainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        patience=patience,
        device=device,
        seed=seed,
        validation_seed=validation_seed,
        validation_rounds=validation_rounds,
        scheduler_patience=scheduler_patience,
        train_generator=train_generator,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_warmup_epochs=loss_warmup_epochs,
        validation_component_weights=(
            FIXED_VALIDATION_COMPONENT_WEIGHTS
        ),
        checkpoint_metadata={
            "scaler_state": prepared.scaler.state_dict(),
            "calibrated_threshold": 0.0,
            "seed": seed,
            "resolved_config": resolved_config,
            "git_revision": git_revision,
            "source_fingerprint": source_fingerprint,
            "reproducibility": reproducibility,
        },
    )
    start_epoch = 1
    resume_path = getattr(args, "resume", None)
    if resume_path:
        recovery = trainer.load_checkpoint(
            resume_path,
            restore_rng=True,
        )
        if recovery["resolved_config"] != _json_safe(resolved_config):
            raise ValueError(
                "Resume checkpoint resolved_config does not match the "
                "current training invocation"
            )
        start_epoch = trainer.last_epoch + 1
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        save_dir=save_dir,
        save_best=True,
        start_epoch=start_epoch,
    )

    calibration_loaders = [
        DataLoader(
            prepared.train_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    ]
    calibration_scores = collect_calibration_scores(
        trainer.model,
        calibration_loaders,
        device=trainer.device,
    )
    calibrated_threshold = calibrate_threshold(
        calibration_scores,
        percentile=calibration_percentile,
    )
    resolved_config["calibration"]["threshold"] = calibrated_threshold
    metadata_updates = {
        "calibrated_threshold": calibrated_threshold,
        "resolved_config": resolved_config,
    }
    trainer.save_best_checkpoint(
        os.path.join(save_dir, "best_model.pt"),
        metadata_updates=metadata_updates,
    )
    trainer.save_best_checkpoint(
        os.path.join(save_dir, "final_model.pt"),
        metadata_updates=metadata_updates,
    )
    _write_run_config(run_config_path, resolved_config)
    print(
        f"Training complete. Calibrated threshold="
        f"{calibrated_threshold:.6f}. Artifacts saved to {save_dir}"
    )


def detect(args) -> None:
    """Detect anomalies using the checkpoint-calibrated threshold."""
    print("=" * 70)
    print("  SpecMoE-AD Anomaly Detection")
    print("=" * 70)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = load_model_checkpoint(args.model, device=device)
    model_config = checkpoint["model_config"]

    if args.data.endswith(".csv"):
        data = load_csv_data(args.data)
    elif args.data.endswith(".npy"):
        data = load_npy_data(args.data)
    else:
        raise ValueError(
            f"Unsupported format: {args.data}. Use .csv or .npy"
        )
    expected_variables = int(model_config["num_vars"])
    if data.shape[1] != expected_variables:
        raise ValueError(
            f"Checkpoint expects {expected_variables} variables, "
            f"but detection data has {data.shape[1]}"
        )

    scaler = StandardScaler()
    scaler.load_state_dict(checkpoint["scaler_state"])
    window_size = int(model_config["time_len"])
    if data.shape[0] < window_size:
        raise ValueError(
            f"Detection series has {data.shape[0]} points, fewer than "
            f"window_size={window_size}"
        )
    stride = _validated_positive_integer(args.stride, name="stride")
    if stride > window_size:
        raise ValueError(
            "Detection stride cannot exceed the model window size"
        )
    dataset = TimeSeriesDataset(
        data,
        window_size=window_size,
        stride=stride,
        scaler=scaler,
        fit_scaler=False,
        cover_end=True,
    )
    batch_size = _validated_positive_integer(
        args.batch_size,
        name="batch_size",
    )
    rounds = _validated_positive_integer(
        args.n_rounds,
        name="n_rounds",
    )
    scores = score_single_segment(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        n_rounds=rounds,
        require_full_coverage=True,
    )
    exploratory = bool(getattr(args, "exploratory", False))
    exploratory_percentile = (
        getattr(args, "exploratory_percentile", None)
        if exploratory
        else None
    )
    threshold = resolve_detection_threshold(
        checkpoint,
        target_scores=scores,
        exploratory_percentile=exploratory_percentile,
        exploratory=exploratory,
    )
    predictions = detect_label(scores, threshold)

    output_path = args.output or "anomaly_results.npz"
    np.savez(
        output_path,
        scores=scores,
        predictions=predictions,
        threshold=threshold,
    )
    print(f"Results saved to {output_path}")
    print(f"Resolved threshold: {threshold:.6f}")
    print(
        f"Anomalies detected: {int(predictions.sum())} / "
        f"{predictions.size}"
    )


def audit_protocol(args) -> None:
    """Run the 50/10/40 partition preflight without training or artifacts."""
    config = _load_yaml_config(args.config)
    if not config.get("dataset") or not config.get("data_root"):
        raise ValueError("Protocol preflight requires dataset and data_root")
    window_size = _validated_positive_integer(
        config.get("window_size", 256),
        name="window_size",
    )
    stride = _validated_positive_integer(
        config.get("stride", 64),
        name="stride",
    )
    seed = int(config.get("seed", 42))
    loaded = load_dataset_from_config(config)
    configured_features = config.get("n_features")
    if (
        configured_features is not None
        and int(configured_features) != loaded.feature_dim
    ):
        raise ValueError(
            f"Config declares n_features={configured_features}, but "
            f"loaded data has {loaded.feature_dim} variables"
        )
    protocol_split = _build_training_protocol(
        loaded,
        window_size=window_size,
    )
    prepared = prepare_training_data(
        protocol_split,
        window_size=window_size,
        stride=stride,
        seed=seed,
    )
    manifest = protocol_split.manifest
    print(f"Protocol: {manifest['protocol_version']}  seed={seed}  allocation=5:1:4")
    print(
        f"Normal train windows={len(prepared.train_dataset)}  "
        f"validation windows={len(prepared.val_dataset)}  "
        f"test segments={len(protocol_split.test_segments)}"
    )
    for source_id, record in manifest["sources"].items():
        print(
            f"source {source_id}: train={record['train_interval']} "
            f"validation={record['validation_interval']} "
            f"test={record['test_interval']}"
        )



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SpecMoE-AD: Spectrum-Aware Mixture-of-Experts for TSAD"
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    parser_train = subparsers.add_parser(
        "train",
        help="Train a model",
    )
    source_group = parser_train.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", type=str, default=None)
    source_group.add_argument("--data", type=str, default=None)
    parser_train.add_argument("--labels", type=str, default=None)
    parser_train.add_argument("--window_size", type=int, default=None)
    parser_train.add_argument("--stride", type=int, default=None)
    parser_train.add_argument("--num_epochs", type=int, default=None)
    parser_train.add_argument("--batch_size", type=int, default=None)
    parser_train.add_argument("--lr", type=float, default=None)
    parser_train.add_argument("--weight_decay", type=float, default=None)
    parser_train.add_argument("--grad_clip", type=float, default=None)
    parser_train.add_argument("--patience", type=int, default=None)
    parser_train.add_argument(
        "--scheduler-patience",
        dest="scheduler_patience",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--warmup-epochs",
        dest="warmup_epochs",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--warmup-start-factor",
        dest="warmup_start_factor",
        type=float,
        default=None,
    )
    parser_train.add_argument(
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--loss-warmup-epochs",
        dest="loss_warmup_epochs",
        type=int,
        default=None,
    )
    parser_train.add_argument("--seed", type=int, default=None)
    parser_train.add_argument(
        "--validation-seed",
        dest="validation_seed",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--validation-rounds",
        dest="validation_rounds",
        type=int,
        default=None,
    )
    parser_train.add_argument(
        "--calibration-percentile",
        dest="calibration_percentile",
        type=float,
        default=None,
    )
    parser_train.add_argument("--device", type=str, default=None)
    parser_train.add_argument("--save_dir", type=str, default=None)
    parser_train.add_argument("--resume", type=str, default=None)
    parser_train.add_argument("--patch_size", type=int, default=None)
    parser_train.add_argument("--patch_stride", type=int, default=None)
    parser_train.add_argument("--d_model", type=int, default=None)
    parser_train.add_argument("--d_ff", type=int, default=None)
    parser_train.add_argument("--cross_var_heads", type=int, default=None)
    parser_train.add_argument(
        "--use-revin",
        dest="use_revin",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser_train.add_argument(
        "--router-mode",
        dest="router_mode",
        choices=["global", "variable"],
        default=None,
    )
    parser_train.add_argument(
        "--expert-specialization",
        dest="expert_specialization",
        choices=["none", "frequency_bands"],
        default=None,
    )
    parser_train.add_argument("--lambda_freq", type=float, default=None)
    parser_train.add_argument(
        "--score_freq_weight",
        type=float,
        default=None,
    )
    parser_train.add_argument(
        "--disagreement_weight",
        type=float,
        default=None,
    )
    parser_train.add_argument(
        "--input_mask_ratio",
        type=float,
        default=None,
    )
    parser_train.add_argument(
        "--mask-mode",
        dest="mask_mode",
        choices=["random", "contiguous"],
        default=None,
    )
    parser_train.add_argument(
        "--diversity-mode",
        dest="diversity_mode",
        choices=["none", "cosine", "contrastive"],
        default=None,
    )
    parser_train.add_argument(
        "--use_grad_score",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser_train.add_argument(
        "--use_adaptive_patches",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser_train.add_argument(
        "--use_learnable_temp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser_train.add_argument(
        "--frequency-score-aggregation",
        dest="frequency_score_aggregation",
        choices=["mean", "max"],
        default=None,
    )

    parser_detect = subparsers.add_parser(
        "detect",
        help="Detect anomalies",
    )
    parser_detect.add_argument("--data", type=str, required=True)
    parser_detect.add_argument("--model", type=str, required=True)
    parser_detect.add_argument("--output", type=str, default=None)
    parser_detect.add_argument("--stride", type=int, default=1)
    parser_detect.add_argument("--batch_size", type=int, default=16)
    parser_detect.add_argument("--device", type=str, default=None)
    parser_detect.add_argument("--n_rounds", type=int, default=1)
    parser_detect.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow target-distribution percentile thresholding",
    )
    parser_detect.add_argument(
        "--exploratory-percentile",
        dest="exploratory_percentile",
        type=float,
        default=None,
        help="Target percentile used only with --exploratory",
    )

    parser_protocol = subparsers.add_parser(
        "protocol",
        help="Validate and print the unified entity-split protocol",
    )
    parser_protocol.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "detect":
        detect(args)
    elif args.command == "protocol":
        audit_protocol(args)


if __name__ == "__main__":
    main()
