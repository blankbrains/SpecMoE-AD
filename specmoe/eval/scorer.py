"""Thresholding utilities for point-level anomaly scores."""

from typing import Optional

import numpy as np


def _validated_scores(anomaly_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(anomaly_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("anomaly_scores cannot be empty")
    if not np.isfinite(scores).all():
        raise FloatingPointError(
            "anomaly_scores contains NaN/Inf values"
        )
    return scores


def _validated_percentile(percentile: float) -> float:
    if (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, (int, float, np.number))
        or not np.isfinite(float(percentile))
        or not 0.0 < float(percentile) < 100.0
    ):
        raise ValueError("percentile must be finite and in (0, 100)")
    return float(percentile)


def calibrate_threshold(
    normal_reference_scores: np.ndarray,
    *,
    percentile: float,
) -> float:
    """Calibrate an operational threshold from normal reference scores."""
    scores = _validated_scores(normal_reference_scores)
    resolved_percentile = _validated_percentile(percentile)
    return float(np.percentile(scores, resolved_percentile))


def resolve_detection_threshold(
    checkpoint: dict,
    *,
    target_scores: Optional[np.ndarray] = None,
    exploratory_percentile: Optional[float] = None,
    exploratory: bool = False,
) -> float:
    """Resolve the saved threshold unless target calibration is explicit."""
    if type(exploratory) is not bool:
        raise ValueError("exploratory must be boolean")
    if exploratory_percentile is not None and not exploratory:
        raise ValueError(
            "Target percentile thresholding requires exploratory mode"
        )
    if exploratory:
        if target_scores is None:
            raise ValueError(
                "Exploratory thresholding requires target_scores"
            )
        if exploratory_percentile is None:
            raise ValueError(
                "Exploratory thresholding requires "
                "exploratory_percentile"
            )
        return calibrate_threshold(
            target_scores,
            percentile=exploratory_percentile,
        )

    threshold = checkpoint.get("calibrated_threshold")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.number))
        or not np.isfinite(float(threshold))
    ):
        raise ValueError(
            "Checkpoint calibrated_threshold must be a finite number"
        )
    return float(threshold)


def threshold_search(
    anomaly_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    percentile: float = 95.0,
    return_metrics: bool = False,
):
    """Choose a percentile threshold or tune one on a designated labeled split."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    scores = _validated_scores(anomaly_scores)

    if labels is None:
        return float(np.percentile(scores, percentile))

    labels_array = np.asarray(labels)
    if not (
        np.issubdtype(labels_array.dtype, np.number)
        or np.issubdtype(labels_array.dtype, np.bool_)
    ):
        raise ValueError("labels must contain binary numeric values")
    if not np.isfinite(labels_array).all():
        raise FloatingPointError("labels contains NaN/Inf values")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("labels must contain only binary 0/1 values")
    labels_flat = labels_array.ravel().astype(bool, copy=False)
    if labels_flat.shape != scores.shape:
        raise ValueError("labels and anomaly_scores must contain the same number of values")

    best_threshold = float(np.percentile(scores, percentile))
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    unique_scores = np.unique(scores)
    candidates = np.concatenate(
        (
            np.array(
                [np.nextafter(unique_scores[0], -np.inf)],
                dtype=np.float64,
            ),
            unique_scores,
        )
    )
    for threshold in candidates:
        prediction = scores > threshold
        tp = int((prediction & labels_flat).sum())
        fp = int((prediction & ~labels_flat).sum())
        fn = int((~prediction & labels_flat).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}

    if return_metrics:
        return best_threshold, best_metrics
    return best_threshold


def detect_label(anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert anomaly scores into binary point labels."""
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.number))
        or not np.isfinite(float(threshold))
    ):
        raise ValueError("threshold must be a finite number")
    scores = np.asarray(anomaly_scores)
    if not np.isfinite(scores).all():
        raise FloatingPointError("anomaly_scores contains NaN/Inf values")
    return (scores > threshold).astype(np.int32)
