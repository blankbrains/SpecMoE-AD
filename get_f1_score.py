"""Exact point and point-adjusted F1 metrics."""

from __future__ import annotations

import numpy as np


def _validate_inputs(score, label):
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel()
    if score.size == 0:
        raise ValueError("Scores cannot be empty")
    if score.shape != label.shape:
        raise ValueError("Scores and labels must have the same shape")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    return score, label.astype(bool, copy=False)


def calc_p2p(predict, actual):
    """Return F1, precision, recall, and confusion counts without epsilons."""
    predict = np.asarray(predict, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    if predict.shape != actual.shape:
        raise ValueError("Prediction and label shapes must match")
    tp = int(np.sum(predict & actual))
    tn = int(np.sum(~predict & ~actual))
    fp = int(np.sum(predict & ~actual))
    fn = int(np.sum(~predict & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return f1, precision, recall, tp, tn, fp, fn


def point_adjust(score, label, thres):
    """Propagate a hit through the complete ground-truth anomaly range."""
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel()
    if score.shape != label.shape:
        raise ValueError("Scores and labels must have the same shape")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")

    prediction = score >= float(thres)
    actual = label.astype(bool, copy=False)
    padded = np.concatenate(([False], actual, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, end in zip(starts, ends):
        if prediction[start:end].any():
            prediction[start:end] = True
    return prediction, actual


def point_adjust_segments(scores, labels, threshold):
    """Apply point adjustment independently to each independent segment."""
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must contain the same segments")
    predictions = []
    actuals = []
    for score, label in zip(scores, labels):
        prediction, actual = point_adjust(score, label, threshold)
        predictions.append(prediction)
        actuals.append(actual)
    return predictions, actuals


def _threshold_candidates(score):
    unique = np.unique(score)
    return np.concatenate(
        (
            [np.nextafter(unique[-1], np.inf)],
            unique[::-1],
            [np.nextafter(unique[0], -np.inf)],
        )
    )


def best_f1_without_pointadjust(score, label):
    """Search every unique score boundary for exact point-wise F1."""
    score, label = _validate_inputs(score, label)
    best = (-1.0, 0.0, 0.0, None, None)
    for threshold in _threshold_candidates(score):
        prediction = score >= threshold
        f1, precision, recall, *_ = calc_p2p(prediction, label)
        if f1 > best[0]:
            best = (f1, precision, recall, threshold, prediction)
    return best[0], best[1], best[2], best[4]


def best_f1(score, label):
    """Search every unique score boundary for point-adjusted F1."""
    score, label = _validate_inputs(score, label)
    best = (-1.0, 0.0, 0.0, None, None)
    for threshold in _threshold_candidates(score):
        prediction, actual = point_adjust(score, label, threshold)
        f1, precision, recall, *_ = calc_p2p(prediction, actual)
        if f1 > best[0]:
            best = (f1, precision, recall, threshold, prediction)
    return best[0], best[1], best[2], best[4]
