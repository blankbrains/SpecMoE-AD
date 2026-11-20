"""
Preprocessing utilities for SpecMoE-AD.

Implements StandardScaler for dataset-level normalization and RevIN
(Reversible Instance Normalization) for per-sample normalization before
frequency-domain transformation.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from specmoe.data.types import TimeSeriesSegment


_SCALER_ROW_CHUNK_SIZE = 65_536


class StandardScaler:
    """
    Dataset-level standard scaling: fits mean/std on training data,
    applies (x - mean) / (std + eps) during transform.

    Stores statistics for inverse transform to recover original scale.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability.
        """
        try:
            eps = float(eps)
        except (TypeError, ValueError) as exc:
            raise ValueError("StandardScaler eps must be finite and positive") from exc
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("StandardScaler eps must be finite and positive")
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps
        self._fitted = False

    @staticmethod
    def _arrays(data) -> Tuple[list, str]:
        if isinstance(data, TimeSeriesSegment):
            return [data.values], "segment"
        if isinstance(data, np.ndarray):
            array = np.asarray(data)
            if array.ndim == 2:
                return [array], "array2d"
            if array.ndim == 3:
                return [array[index] for index in range(array.shape[0])], "array3d"
            raise ValueError(
                f"Expected 2D/3D array or segments, got shape {array.shape}"
            )
        try:
            segments = list(data)
        except TypeError as exc:
            raise ValueError(
                "Expected a numeric array or sequence of TimeSeriesSegment"
            ) from exc
        if not segments or not all(
            isinstance(segment, TimeSeriesSegment) for segment in segments
        ):
            raise ValueError(
                "Segment scaling requires a non-empty TimeSeriesSegment sequence"
            )
        return [segment.values for segment in segments], "segments"

    @staticmethod
    def _validate_arrays(arrays: Sequence[np.ndarray]) -> int:
        feature_dims = set()
        for array in arrays:
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(
                    "StandardScaler inputs must have non-empty [time, feature] shape"
                )
            if not np.isfinite(array).all():
                invalid_count = int((~np.isfinite(array)).sum())
                raise ValueError(
                    "StandardScaler input contains "
                    f"{invalid_count} non-finite NaN/Inf values"
                )
            feature_dims.add(int(array.shape[1]))
        if len(feature_dims) != 1:
            raise ValueError(
                f"StandardScaler segment feature widths differ: {sorted(feature_dims)}"
            )
        return feature_dims.pop()

    def fit(self, data) -> "StandardScaler":
        """Fit per-feature statistics across arrays or independent segments."""
        arrays, _ = self._arrays(data)
        return self._fit_arrays_online(arrays)

    def fit_segments(
        self,
        segments: Sequence[TimeSeriesSegment],
    ) -> "StandardScaler":
        """Fit across segments using cumulative population statistics."""
        arrays, kind = self._arrays(segments)
        if kind != "segments":
            raise ValueError("fit_segments requires TimeSeriesSegment inputs")
        return self._fit_arrays_online(arrays)

    def _fit_arrays_online(
        self,
        arrays: Sequence[np.ndarray],
    ) -> "StandardScaler":
        feature_dim = self._validate_arrays(arrays)
        count = 0
        mean = np.zeros(feature_dim, dtype=np.float64)
        squared_deviation = np.zeros(feature_dim, dtype=np.float64)
        for array in arrays:
            for start in range(0, array.shape[0], _SCALER_ROW_CHUNK_SIZE):
                stop = min(start + _SCALER_ROW_CHUNK_SIZE, array.shape[0])
                batch = np.asarray(array[start:stop], dtype=np.float64)
                batch_count = int(batch.shape[0])
                batch_mean = batch.mean(axis=0)
                centered = batch - batch_mean
                batch_squared_deviation = np.sum(centered * centered, axis=0)
                if count == 0:
                    mean = batch_mean
                    squared_deviation = batch_squared_deviation
                    count = batch_count
                    continue
                combined_count = count + batch_count
                delta = batch_mean - mean
                mean = mean + delta * (batch_count / combined_count)
                squared_deviation = (
                    squared_deviation
                    + batch_squared_deviation
                    + delta * delta * (count * batch_count / combined_count)
                )
                count = combined_count
        self.mean = mean.reshape(1, 1, feature_dim)
        self.std = np.sqrt(
            squared_deviation / count
        ).reshape(1, 1, feature_dim)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)
        self._fitted = True
        return self

    def _validate_fitted_for_width(self, feature_dim: int) -> None:
        if not self._fitted:
            raise RuntimeError("StandardScaler must be fit before transform.")
        if self.mean is None or self.std is None:
            raise ValueError("Fitted StandardScaler is missing mean/std state")
        expected_shape = (1, 1, feature_dim)
        if self.mean.shape != expected_shape or self.std.shape != expected_shape:
            raise ValueError(
                "StandardScaler state feature shape is incompatible: expected "
                f"{expected_shape}, got mean={self.mean.shape}, std={self.std.shape}"
            )

    def _transform_array(self, array: np.ndarray, inverse: bool = False) -> np.ndarray:
        original = np.asarray(array)
        if original.ndim not in (2, 3):
            raise ValueError(
                f"StandardScaler transform expects 2D/3D data, got {original.shape}"
            )
        if not np.isfinite(original).all():
            invalid_count = int((~np.isfinite(original)).sum())
            raise ValueError(
                f"Cannot transform data containing {invalid_count} non-finite NaN/Inf values"
            )
        self._validate_fitted_for_width(int(original.shape[-1]))
        if inverse:
            result = original * (self.std + self.eps) + self.mean
        else:
            result = (original - self.mean) / (self.std + self.eps)
        if original.ndim == 2:
            result = result[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("StandardScaler produced non-finite values")
        return result.astype(np.float32, copy=False)

    def transform(self, data):
        """Transform an array, one segment, or a segment sequence."""
        if isinstance(data, TimeSeriesSegment):
            return TimeSeriesSegment(
                data.segment_id,
                self._transform_array(data.values),
                data.labels,
                data.metadata,
            )
        if isinstance(data, np.ndarray):
            return self._transform_array(data)
        segments = list(data)
        if not segments or not all(
            isinstance(segment, TimeSeriesSegment) for segment in segments
        ):
            raise ValueError(
                "StandardScaler segment transform requires TimeSeriesSegment values"
            )
        return [
            TimeSeriesSegment(
                segment.segment_id,
                self._transform_array(segment.values),
                segment.labels,
                segment.metadata,
            )
            for segment in segments
        ]

    def inverse_transform(self, data):
        """Reverse normalization for arrays or segments."""
        if isinstance(data, TimeSeriesSegment):
            return TimeSeriesSegment(
                data.segment_id,
                self._transform_array(data.values, inverse=True),
                data.labels,
                data.metadata,
            )
        if isinstance(data, np.ndarray):
            return self._transform_array(data, inverse=True)
        segments = list(data)
        return [
            TimeSeriesSegment(
                segment.segment_id,
                self._transform_array(segment.values, inverse=True),
                segment.labels,
                segment.metadata,
            )
            for segment in segments
        ]

    def fit_transform(self, data):
        """Fit and transform in one call."""
        return self.fit(data).transform(data)

    def state_dict(self) -> dict:
        """Serialize scaler state."""
        return {
            "mean": None if self.mean is None else self.mean.copy(),
            "std": None if self.std is None else self.std.copy(),
            "eps": self.eps,
            "_fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scaler state."""
        if not isinstance(state, dict):
            raise ValueError("StandardScaler state must be a dictionary")
        required_fields = {"mean", "std", "eps", "_fitted"}
        actual_fields = set(state)
        if actual_fields != required_fields:
            missing = sorted(required_fields - actual_fields)
            extra = sorted(actual_fields - required_fields)
            raise ValueError(
                "StandardScaler state schema mismatch: "
                f"missing fields={missing}, extra fields={extra}"
            )
        fitted = state["_fitted"]
        if not isinstance(fitted, (bool, np.bool_)):
            raise ValueError("StandardScaler _fitted state must be boolean")
        eps = float(state["eps"])
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("StandardScaler eps must be finite and positive")
        if not fitted:
            if state.get("mean") is not None or state.get("std") is not None:
                raise ValueError("Unfitted StandardScaler state cannot contain mean/std")
            self.mean = None
            self.std = None
            self.eps = eps
            self._fitted = False
            return
        if "mean" not in state or "std" not in state:
            raise ValueError("Fitted StandardScaler state requires mean and std")
        def _as_cpu_array(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        mean = _as_cpu_array(state["mean"])
        std = _as_cpu_array(state["std"])
        if (
            mean.ndim != 3
            or std.ndim != 3
            or mean.shape != std.shape
            or mean.shape[:2] != (1, 1)
            or mean.shape[2] < 1
        ):
            raise ValueError(
                "StandardScaler mean/std shape must be [1, 1, feature_width]"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("StandardScaler mean/std must be finite")
        if np.any(std <= 0):
            raise ValueError("StandardScaler std must be strictly positive")
        self.mean = mean.astype(np.float32, copy=True)
        self.std = std.astype(np.float32, copy=True)
        self.eps = eps
        self._fitted = True


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2022).

    Normalizes each sample along the time dimension so that the per-instance
    distribution has zero mean and unit variance. Stores the statistics for
    exact reversal (denormalization) after reconstruction.

    Applied BEFORE FFT to stabilize spectral decomposition across samples
    with varying scales.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Args:
            num_features: Number of variables (N).
            eps: Small constant for numerical stability.
            affine: If True, learn affine parameters gamma/beta applied
                    after normalization.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            # [1, 1, N] for broadcasting over [B, T, N]
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        # Buffers to store per-sample statistics for denormalization
        self.register_buffer("_saved_mean", None, persistent=False)
        self.register_buffer("_saved_std", None, persistent=False)

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """Normalize or denormalize.

        Args:
            x: [B, T, N] input tensor.
            mode: 'norm' for normalization, 'denorm' for denormalization.

        Returns:
            Tensor of same shape [B, T, N].
        """
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Instance normalization along time dimension.

        Args:
            x: [B, T, N]

        Returns:
            [B, T, N] normalized.
        """
        # Compute per-sample statistics along time axis (dim=1)
        self._saved_mean = x.mean(dim=1, keepdim=True)  # [B, 1, N]
        self._saved_std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps  # [B, 1, N]
        x = (x - self._saved_mean) / self._saved_std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse instance normalization.

        Args:
            x: [B, T, N]

        Returns:
            [B, T, N] denormalized.
        """
        if self._saved_mean is None or self._saved_std is None:
            raise RuntimeError(
                "RevIN._denormalize called before _normalize. "
                "Call forward(mode='norm') first to save statistics."
            )
        if self.affine:
            x = (x - self.beta) / self.gamma
        x = x * self._saved_std + self._saved_mean
        return x

    def get_saved_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return saved mean and std from last normalization.

        Returns:
            (mean, std) each of shape [B, 1, N].
        """
        if self._saved_mean is None:
            raise RuntimeError("No saved statistics. Call forward(mode='norm') first.")
        return self._saved_mean, self._saved_std
