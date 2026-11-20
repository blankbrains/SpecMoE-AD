"""M²B-MoE fixed-protocol evaluation with the required 13 metrics.

Usage:
  python eval_full.py --checkpoint checkpoints/xxx/best_model.pt --dataset smd

13 指标: P, R, F1, P_PA, R_PA, F1_PA, Aff-F1, AUC-ROC, AUC-PR, R-AUC-ROC, R-AUC-PR, VUS-ROC, VUS-PR
"""
import sys, os, argparse, hashlib, torch, numpy as np, time, json
from collections.abc import Mapping
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from specmoe.model.loading import load_model_checkpoint
from specmoe.data.protocol import (
    OFFICIAL_PROTOCOL_VERSION,
    PAPER_PROTOCOL_VERSION,
    materialize_manifest_test_segments,
    materialize_manifest_validation_segments,
)
from specmoe.data.datasets import DATASET_REGISTRY
from metrics import (
    METRIC_PROTOCOL_REVISION,
    OFFICIAL_VUS_COMMIT,
    OFFICIAL_VUS_REPOSITORY,
    OFFICIAL_VUS_THRESHOLD_COUNT,
    RangeAUC_vectorized,
    RangeAUC_volume,
    convert_vector_to_events,
    pr_from_events,
)
from get_f1_score import (
    best_f1,
    best_f1_without_pointadjust,
    point_adjust_segments,
)


ALL_METRICS = [
    "P", "R", "F1",
    "P_PA", "R_PA", "F1_PA",
    "Aff-F1",
    "AUC-ROC", "AUC-PR",
    "R-AUC-ROC", "R-AUC-PR",
    "VUS-ROC", "VUS-PR",
]
_MAX_STFT_SIGNALS_PER_BATCH = 32_768
_IMP_COMPONENT_KEYS = (
    "anom_score",
    "time_score",
    "freq_score",
    "disagreement",
    "gradient_score",
)
_MULTI_VIEW_RANK_WEIGHTS = {
    "time_score": 0.40,
    "freq_score": 0.30,
    "disagreement": 0.10,
    "gradient_score": 0.20,
}
_MULTI_VIEW_SPECTRAL_RANK_WEIGHTS = {
    "time_score": 0.30,
    "freq_score": 0.25,
    "disagreement": 0.10,
    "gradient_score": 0.15,
    "spectral_residual": 0.20,
}
_MULTI_VIEW_SPECTRAL_UNCERTAINTY_RANK_WEIGHTS = {
    "time_score": 0.25,
    "freq_score": 0.20,
    "disagreement": 0.08,
    "gradient_score": 0.12,
    "spectral_residual": 0.15,
    "mask_uncertainty": 0.20,
}
_TRIMMED_REFERENCE_PERCENTILE = 80.0

DATASET_DIRECTORY_NAMES = {
    "smd": "SMD",
    "msl": "MSL",
    "swat": "SWAT",
    "wadi": "WADI",
    "psm": "PSM",
}


def spectral_residual(x, q=3):
    n = len(x); f = np.fft.fft(x); amp = np.abs(f); phase = np.angle(f)
    L = np.log(amp + 1e-8); k = np.ones(q) / q
    AL = np.convolve(L, k, mode='same'); R = L - AL
    sal = np.abs(np.fft.ifft(np.exp(R + 1j * phase)))
    ks = np.ones(3) / 3; return np.convolve(sal, ks, mode='same')


def compute_affiliation(score, y):
    events_gt = convert_vector_to_events(y)
    if not events_gt:
        raise ValueError("Affiliation metrics require at least one ground-truth event")
    Trange = (0, len(y))
    thresholds = np.unique(np.percentile(score, np.linspace(0, 100, 50)))
    best_f1_val, best_prec, best_rec = 0.0, 0.0, 0.0
    for th in thresholds:
        pred_bin = (score > th).astype(int)
        events_pred = convert_vector_to_events(pred_bin)
        if not events_pred:
            continue
        result = pr_from_events(events_pred, events_gt, Trange)
        prec = float(result.get('Affiliation_Precision', 0) or 0)
        rec = float(result.get('Affiliation_Recall', 0) or 0)
        if not np.isfinite(prec):
            prec = 0.0
        if not np.isfinite(rec):
            rec = 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1_val:
            best_f1_val, best_prec, best_rec = f1, prec, rec
    return float(best_f1_val), float(best_prec), float(best_rec)


def compute_all_metrics(score, y):
    """Compute all 13 metrics for a given score vector."""
    score = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.int32).ravel()
    if score.size == 0:
        raise ValueError("Metric inputs cannot be empty")
    if score.shape != y.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if np.unique(y).size != 2:
        raise ValueError("AUC metrics require both normal and anomalous labels")
    auc_roc = roc_auc_score(y, score)
    auc_pr = average_precision_score(y, score)
    bf1, prec, rec, _ = best_f1_without_pointadjust(score, y)
    bf1_pa, prec_pa, rec_pa, _ = best_f1(score, y)
    aff_f1, _, _ = compute_affiliation(score, y)
    m2 = evaluate_metrics(score, y, slidingWindow=30)
    return {
        "P": float(prec), "R": float(rec), "F1": float(bf1),
        "P_PA": float(prec_pa), "R_PA": float(rec_pa), "F1_PA": float(bf1_pa),
        "Aff-F1": float(aff_f1),
        "AUC-ROC": float(auc_roc), "AUC-PR": float(auc_pr),
        "R-AUC-ROC": float(m2.get('r_auc_roc', 0)), "R-AUC-PR": float(m2.get('r_auc_pr', 0)),
        "VUS-ROC": float(m2.get('vus_roc', 0)), "VUS-PR": float(m2.get('vus_pr', 0)),
    }


def _resolve_imp_batch_size(num_vars):
    """Bound the flattened [windows * variables] STFT batch."""
    if (
        isinstance(num_vars, bool)
        or not isinstance(num_vars, (int, np.integer))
        or int(num_vars) < 1
    ):
        raise ValueError("num_vars must be a positive integer")
    return min(
        1024,
        max(1, _MAX_STFT_SIGNALS_PER_BATCH // int(num_vars)),
    )


def _compute_window_averaged_scores(
    model,
    data_std,
    *,
    W=200,
    n_rounds=3,
    device='cuda',
    seed=42,
    score_keys=("anom_score",),
    return_round_scores=False,
):
    """Aggregate requested model score components over real sliding windows."""
    data_std = np.asarray(data_std, dtype=np.float32)
    if data_std.ndim != 2:
        raise ValueError(f"Expected [time, variables] input, got {data_std.shape}")
    if isinstance(W, bool) or not isinstance(W, (int, np.integer)) or W < 2:
        raise ValueError("W must be an integer of at least two points")
    effective_window = min(int(W), len(data_std))
    if effective_window < 2:
        raise ValueError(
            "Each scored segment must contain at least two real observations"
        )
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    if (
        not isinstance(score_keys, tuple)
        or not score_keys
        or any(type(key) is not str for key in score_keys)
        or len(set(score_keys)) != len(score_keys)
    ):
        raise ValueError("score_keys must be a non-empty tuple of unique strings")
    if type(return_round_scores) is not bool:
        raise TypeError("return_round_scores must be boolean")

    Tv = len(data_std)
    nw = Tv - effective_window + 1
    round_scores = {key: [] for key in score_keys}
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            batch_size = _resolve_imp_batch_size(data_std.shape[1])
            for round_index in range(n_rounds):
                score_sums = {
                    key: np.zeros(Tv, dtype=np.float64)
                    for key in score_keys
                }
                score_count = np.zeros(Tv, dtype=np.float64)
                for i in range(0, nw, batch_size):
                    starts = list(range(i, min(i + batch_size, nw)))
                    x = torch.from_numpy(
                        np.stack(
                            [
                                data_std[
                                    start : start + effective_window
                                ]
                                for start in starts
                            ]
                        )
                    ).to(device)
                    if hasattr(model, "forward_masked"):
                        output = model.forward_masked(
                            x,
                            seed=seed + round_index,
                            compute_scores=True,
                            allow_shorter_time_length=(
                                effective_window < int(
                                    getattr(
                                        model,
                                        "time_len",
                                        effective_window,
                                    )
                                )
                            ),
                        )
                    else:
                        output = model(x)
                    window_scores = {}
                    for key in score_keys:
                        value = output.get(key)
                        if not isinstance(value, torch.Tensor):
                            raise ValueError(
                                f"Model output is missing tensor score {key!r}"
                            )
                        if tuple(value.shape) != (
                            len(starts),
                            effective_window,
                        ):
                            raise ValueError(
                                f"Model score {key!r} must have shape "
                                f"{(len(starts), effective_window)}, got "
                                f"{tuple(value.shape)}"
                            )
                        window_scores[key] = (
                            value.detach().cpu().numpy()
                        )
                    for local_index, start in enumerate(starts):
                        for key, values in window_scores.items():
                            score_sums[key][
                                start : start + effective_window
                            ] += values[local_index]
                        score_count[
                            start : start + effective_window
                        ] += 1.0

                if np.any(score_count == 0):
                    raise RuntimeError(
                        "Sliding-window evaluation left uncovered time points"
                    )
                for key, values in score_sums.items():
                    round_scores[key].append(values / score_count)

    stacked_round_scores = {
        key: np.stack(values, axis=0)
        for key, values in round_scores.items()
    }
    scores = {
        key: values.mean(axis=0)
        for key, values in stacked_round_scores.items()
    }
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise FloatingPointError("Model produced non-finite anomaly scores")
    if not all(
        np.isfinite(values).all()
        for values in stacked_round_scores.values()
    ):
        raise FloatingPointError("Model produced non-finite round scores")
    if return_round_scores:
        return scores, stacked_round_scores
    return scores


def compute_imp_score(model, data_std, W=200, n_rounds=3, device='cuda',
                      seed=42):
    """Compute model anomaly scores with exact sliding-window overlap averaging."""
    return _compute_window_averaged_scores(
        model,
        data_std,
        W=W,
        n_rounds=n_rounds,
        device=device,
        seed=seed,
        score_keys=("anom_score",),
    )["anom_score"]


def compute_imp_score_components(
    model,
    data_std,
    W=200,
    n_rounds=3,
    device='cuda',
    seed=42,
):
    """Return every fixed multi-view model score component per time point."""
    return _compute_window_averaged_scores(
        model,
        data_std,
        W=W,
        n_rounds=n_rounds,
        device=device,
        seed=seed,
        score_keys=_IMP_COMPONENT_KEYS,
    )


def compute_imp_score_components_with_uncertainty(
    model,
    data_std,
    W=200,
    n_rounds=3,
    device='cuda',
    seed=42,
):
    """Return fixed score components plus masked-imputation uncertainty."""
    scores, round_scores = _compute_window_averaged_scores(
        model,
        data_std,
        W=W,
        n_rounds=n_rounds,
        device=device,
        seed=seed,
        score_keys=_IMP_COMPONENT_KEYS,
        return_round_scores=True,
    )
    result = dict(scores)
    result["mask_uncertainty"] = np.var(
        round_scores["anom_score"],
        axis=0,
        ddof=0,
    )
    if not np.isfinite(result["mask_uncertainty"]).all():
        raise FloatingPointError(
            "Masked-imputation uncertainty contains non-finite values"
        )
    return result


def compute_sr_score(data):
    """Spectral Residual per-variable max."""
    T, C = data.shape; srs = np.zeros((C, T))
    for c in range(C): srs[c] = spectral_residual(data[:, c])
    return srs.max(axis=0)


def multi_scale(raw):
    """3-scale smoothing + dilation → max."""
    s1 = np.convolve(raw, np.ones(3) / 3, mode='same')
    s2 = np.convolve(raw, np.ones(7) / 7, mode='same'); s2 = maximum_filter1d(s2, size=11)
    s3 = np.convolve(raw, np.ones(15) / 15, mode='same'); s3 = maximum_filter1d(s3, size=31)
    return np.max(np.stack([s1, s2, s3], axis=0), axis=0)


def _scaler_parameters(scaler_state, num_vars):
    """Validate and unpack the exact scaler saved during training."""
    if not isinstance(scaler_state, Mapping):
        raise ValueError("Checkpoint is missing a valid scaler_state")
    if not scaler_state.get("_fitted", False):
        raise ValueError("Checkpoint scaler_state is not fitted")
    if "mean" not in scaler_state or "std" not in scaler_state:
        raise ValueError("Checkpoint scaler_state must contain mean and std")
    def _as_cpu_array(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    mean = _as_cpu_array(scaler_state["mean"]).reshape(-1)
    std = _as_cpu_array(scaler_state["std"]).reshape(-1)
    eps = float(scaler_state.get("eps", 1e-8))
    if mean.size != num_vars or std.size != num_vars:
        raise ValueError(
            f"Scaler has {mean.size} means and {std.size} standard deviations, "
            f"but the data has {num_vars} variables"
        )
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("Checkpoint scaler epsilon must be finite and positive")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("Checkpoint scaler contains NaN/Inf values")
    if np.any(std < 0):
        raise ValueError("Checkpoint scaler standard deviations cannot be negative")
    return mean, np.where(std < eps, 1.0, std), eps


def standardize_test_data(test_data, scaler_state):
    """Standardize test data with the exact scaler saved during training."""
    test_data = np.asarray(test_data)
    if test_data.ndim != 2:
        raise ValueError(f"Expected [time, variables] test data, got {test_data.shape}")
    mean, std, eps = _scaler_parameters(scaler_state, test_data.shape[1])
    standardized = (test_data - mean) / (std + eps)
    if not np.isfinite(standardized).all():
        raise FloatingPointError("Standardized test data contains NaN/Inf values")
    return standardized.astype(np.float32, copy=False)


def _rank_fuse_score(view_scores, expected_length, weights):
    """Fuse fixed score views after independent rank normalization."""
    if not isinstance(view_scores, Mapping):
        raise TypeError("view_scores must be a mapping")
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    expected_keys = set(weights)
    missing = sorted(expected_keys - set(view_scores))
    if missing:
        raise ValueError(
            "view_scores is missing required views: "
            + ", ".join(missing)
        )
    fused = np.zeros(expected_length, dtype=np.float64)
    for key, weight in weights.items():
        values = np.asarray(view_scores[key], dtype=np.float64).ravel()
        if values.shape != (expected_length,):
            raise ValueError(
                f"view_scores[{key!r}] must have shape "
                f"({expected_length},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"view_scores[{key!r}] contains NaN/Inf values"
            )
        fused += weight * rankdata(values, method="average") / expected_length
    return fused


def _multi_view_rank_score(component_scores, expected_length):
    """Fuse fixed model score views after independent rank normalization."""
    return _rank_fuse_score(
        component_scores,
        expected_length,
        _MULTI_VIEW_RANK_WEIGHTS,
    )


def _multi_view_spectral_rank_score(
    component_scores,
    spectral_residual,
    expected_length,
):
    """Fuse model views with an independent spectral-residual rank view."""
    if spectral_residual is None:
        raise ValueError(
            "The 'mv_sr_rank' strategy requires a spectral-residual score"
        )
    if not isinstance(component_scores, Mapping):
        raise TypeError("component_scores must be a mapping")
    views = dict(component_scores)
    views["spectral_residual"] = spectral_residual
    return _rank_fuse_score(
        views,
        expected_length,
        _MULTI_VIEW_SPECTRAL_RANK_WEIGHTS,
    )


def _multi_view_spectral_uncertainty_rank_score(
    component_scores,
    spectral_residual,
    expected_length,
):
    """Fuse fixed model, spectral, and masked-imputation uncertainty views."""
    if spectral_residual is None:
        raise ValueError(
            "The 'mv_sr_unc_rank' strategy requires a spectral-residual score"
        )
    if not isinstance(component_scores, Mapping):
        raise TypeError("component_scores must be a mapping")
    views = dict(component_scores)
    views["spectral_residual"] = spectral_residual
    return _rank_fuse_score(
        views,
        expected_length,
        _MULTI_VIEW_SPECTRAL_UNCERTAINTY_RANK_WEIGHTS,
    )


def _robust_reference_transform(values, reference_values):
    """Map scores to stable units using unlabeled reference observations."""
    values = np.asarray(values, dtype=np.float64).ravel()
    reference_values = np.asarray(
        reference_values, dtype=np.float64
    ).ravel()
    if values.size < 1 or reference_values.size < 1:
        raise ValueError("Score and reference values must be non-empty")
    if not np.isfinite(values).all() or not np.isfinite(reference_values).all():
        raise FloatingPointError(
            "Score and reference values must be finite"
        )
    center = float(np.median(reference_values))
    scale = float(
        1.4826 * np.median(np.abs(reference_values - center))
    )
    if scale <= 1e-12:
        scale = float(np.std(reference_values))
    if scale <= 1e-12:
        return np.zeros(values.shape, dtype=np.float64)
    return np.arcsinh((values - center) / scale)


def _trimmed_robust_reference_transform(values, reference_values):
    """Calibrate against the lower fixed percentile of an unlabeled reference."""
    values = np.asarray(values, dtype=np.float64).ravel()
    reference_values = np.asarray(
        reference_values, dtype=np.float64
    ).ravel()
    if values.size < 1 or reference_values.size < 1:
        raise ValueError("Score and reference values must be non-empty")
    if not np.isfinite(values).all() or not np.isfinite(reference_values).all():
        raise FloatingPointError(
            "Score and reference values must be finite"
        )
    cutoff = float(
        np.percentile(reference_values, _TRIMMED_REFERENCE_PERCENTILE)
    )
    reference_core = reference_values[reference_values <= cutoff]
    if reference_core.size < 1:
        raise RuntimeError("Trimmed reference unexpectedly contains no values")
    return _robust_reference_transform(values, reference_core)


def _validation_calibrated_multi_view_score(
    component_scores,
    spectral_residual,
    validation_components,
    validation_spectral_residual,
    expected_length,
):
    """Fuse five score views after entity-local robust validation calibration."""
    if spectral_residual is None or validation_spectral_residual is None:
        raise ValueError(
            "Validation-calibrated fusion requires spectral-residual scores"
        )
    if not isinstance(component_scores, Mapping) or not isinstance(
        validation_components, Mapping
    ):
        raise TypeError("Model score components must be mappings")
    views = dict(component_scores)
    references = dict(validation_components)
    views["spectral_residual"] = spectral_residual
    references["spectral_residual"] = validation_spectral_residual
    expected_keys = set(_MULTI_VIEW_SPECTRAL_RANK_WEIGHTS)
    missing_views = sorted(expected_keys - set(views))
    missing_references = sorted(expected_keys - set(references))
    if missing_views or missing_references:
        raise ValueError(
            "Validation-calibrated fusion is missing required score views: "
            + ", ".join(missing_views + missing_references)
        )
    fused = np.zeros(expected_length, dtype=np.float64)
    for key, weight in _MULTI_VIEW_SPECTRAL_RANK_WEIGHTS.items():
        values = np.asarray(views[key], dtype=np.float64).ravel()
        if values.shape != (expected_length,):
            raise ValueError(
                f"Score view {key!r} must have shape ({expected_length},), "
                f"got {values.shape}"
            )
        fused += weight * _robust_reference_transform(
            values,
            references[key],
        )
    return fused


def _trimmed_validation_calibrated_multi_view_score(
    component_scores,
    spectral_residual,
    validation_components,
    validation_spectral_residual,
    expected_length,
):
    """Fuse five views using a tail-contamination-resistant reference."""
    if spectral_residual is None or validation_spectral_residual is None:
        raise ValueError(
            "Trimmed validation calibration requires spectral-residual scores"
        )
    if not isinstance(component_scores, Mapping) or not isinstance(
        validation_components, Mapping
    ):
        raise TypeError("Model score components must be mappings")
    views = dict(component_scores)
    references = dict(validation_components)
    views["spectral_residual"] = spectral_residual
    references["spectral_residual"] = validation_spectral_residual
    expected_keys = set(_MULTI_VIEW_SPECTRAL_RANK_WEIGHTS)
    missing_views = sorted(expected_keys - set(views))
    missing_references = sorted(expected_keys - set(references))
    if missing_views or missing_references:
        raise ValueError(
            "Trimmed validation calibration is missing required score views: "
            + ", ".join(missing_views + missing_references)
        )
    fused = np.zeros(expected_length, dtype=np.float64)
    for key, weight in _MULTI_VIEW_SPECTRAL_RANK_WEIGHTS.items():
        values = np.asarray(views[key], dtype=np.float64).ravel()
        if values.shape != (expected_length,):
            raise ValueError(
                f"Score view {key!r} must have shape ({expected_length},), "
                f"got {values.shape}"
            )
        fused += weight * _trimmed_robust_reference_transform(
            values,
            references[key],
        )
    return fused


def select_strategy(
    imp_raw,
    sr_raw,
    strategy="imp",
    fuse_alpha=0.3,
    dilation=0,
    component_scores=None,
    validation_components=None,
    validation_spectral_residual=None,
):
    """Apply one predeclared scoring strategy without consulting test labels."""
    imp_raw = np.asarray(imp_raw, dtype=np.float64).ravel()
    if imp_raw.size == 0 or not np.isfinite(imp_raw).all():
        raise ValueError("imp_raw must contain finite values")
    if strategy == "imp":
        score = imp_raw.copy()
    elif strategy == "sr":
        if sr_raw is None:
            raise ValueError("The 'sr' strategy requires a spectral-residual score")
        score = sr_raw.copy()
    elif strategy in {"fuse", "ms_fuse"}:
        if sr_raw is None:
            raise ValueError("Fusion requires a spectral-residual score")
        imp_rank = rankdata(imp_raw) / len(imp_raw)
        sr_rank = rankdata(sr_raw) / len(sr_raw)
        score = fuse_alpha * sr_rank + (1.0 - fuse_alpha) * imp_rank
    elif strategy == "ms_imp":
        score = multi_scale(imp_raw)
    elif strategy == "ms_sr":
        if sr_raw is None:
            raise ValueError("The 'ms_sr' strategy requires a spectral-residual score")
        score = multi_scale(sr_raw)
    elif strategy == "mv_rank":
        score = _multi_view_rank_score(component_scores, len(imp_raw))
    elif strategy == "mv_sr_rank":
        score = _multi_view_spectral_rank_score(
            component_scores,
            sr_raw,
            len(imp_raw),
        )
    elif strategy == "mv_sr_unc_rank":
        score = _multi_view_spectral_uncertainty_rank_score(
            component_scores,
            sr_raw,
            len(imp_raw),
        )
    elif strategy == "mv_sr_valcal":
        score = _validation_calibrated_multi_view_score(
            component_scores,
            sr_raw,
            validation_components,
            validation_spectral_residual,
            len(imp_raw),
        )
    elif strategy == "mv_sr_trimcal":
        score = _trimmed_validation_calibrated_multi_view_score(
            component_scores,
            sr_raw,
            validation_components,
            validation_spectral_residual,
            len(imp_raw),
        )
    else:
        raise ValueError(f"Unknown evaluation strategy: {strategy}")

    if strategy == "ms_fuse":
        score = multi_scale(score)
    if dilation > 0:
        score = maximum_filter1d(score, size=dilation)
    return score


def evaluate_single(model, loader_fn, data_root, device='cuda', strategy='imp',
                    fuse_alpha=0.3, dilation=0, n_rounds=3, scaler_state=None):
    """Evaluate one checkpoint using a fixed, predeclared strategy."""
    _, te, _, y = loader_fn(data_root)
    y = np.asarray(y).astype(int).ravel()
    te_std = standardize_test_data(te, scaler_state)

    sr_raw = compute_sr_score(te) if strategy in {'sr', 'fuse', 'ms_sr', 'ms_fuse'} else None

    # IMP score
    t0 = time.time()
    W = model.time_len
    imp_raw = compute_imp_score(model, te_std, W=W, n_rounds=n_rounds, device=device)
    imp_time = time.time() - t0
    print(f"  IMP: {imp_time:.0f}s  AUC={roc_auc_score(y, imp_raw):.4f}")

    score = select_strategy(
        imp_raw, sr_raw, strategy=strategy, fuse_alpha=fuse_alpha, dilation=dilation
    )
    label = strategy.upper()
    if strategy in {'fuse', 'ms_fuse'}:
        label += f"_{fuse_alpha:.2f}"
    if dilation > 0:
        label += f"_d{dilation}"
    return label, compute_all_metrics(score, y)


def evaluate_ucr_chunked(model, loader_fn, data_root, device='cuda', chunk_size=500000,
                         strategy='imp', fuse_alpha=0.3, dilation=0, n_rounds=3,
                         scaler_state=None):
    """Evaluate UCR in overlapping chunks while computing metrics globally."""
    _, te, _, y = loader_fn(data_root)
    y = np.asarray(y).astype(int).ravel()
    mean, std, eps = _scaler_parameters(scaler_state, te.shape[1])

    all_scores = []
    for start in range(0, len(te), chunk_size):
        end = min(start + chunk_size, len(te))
        W = model.time_len
        context_start = max(0, start - W + 1)
        context_end = min(len(te), end + W - 1)
        te_context = (te[context_start:context_end] - mean) / (std + eps)
        context_score = compute_imp_score(
            model, te_context, W=W, n_rounds=n_rounds, device=device
        )
        core_start = start - context_start
        core_end = core_start + (end - start)
        all_scores.append(context_score[core_start:core_end])
        print(f"  UCR chunk {start // chunk_size + 1}: {start}-{end} done")

    score = np.concatenate(all_scores)
    sr_raw = None
    if strategy in {'sr', 'fuse', 'ms_sr', 'ms_fuse'}:
        sr_chunks = []
        for start in range(0, len(te), chunk_size):
            end = min(start + chunk_size, len(te))
            sr_chunks.append(compute_sr_score(te[start:end]))
        sr_raw = np.concatenate(sr_chunks)

    selected = select_strategy(
        score, sr_raw, strategy=strategy, fuse_alpha=fuse_alpha, dilation=dilation
    )
    label = strategy.upper()
    if strategy in {'fuse', 'ms_fuse'}:
        label += f"_{fuse_alpha:.2f}"
    if dilation > 0:
        label += f"_d{dilation}"
    return label, compute_all_metrics(selected, y)


def compute_sr_score_segments(segments):
    """Compute spectral-residual scores independently for each segment."""
    return [compute_sr_score(np.asarray(segment)) for segment in segments]


def _affiliation_details_from_prediction(prediction, y):
    """Compute official event-level affiliation details for one segment."""
    prediction = np.asarray(prediction, dtype=np.int32).ravel()
    y = np.asarray(y).ravel()
    if prediction.shape != y.shape:
        raise ValueError("Predictions and labels must have the same length")
    events_gt = convert_vector_to_events(y)
    if not events_gt:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "individual_precision_probabilities": [],
            "individual_recall_probabilities": [],
        }
    events_pred = convert_vector_to_events(prediction)
    result = pr_from_events(events_pred, events_gt, (0, len(y)))
    precision = float(
        result.get(
            "precision",
            result.get("Affiliation_Precision", 0.0),
        )
        or 0.0
    )
    recall = float(
        result.get(
            "recall",
            result.get("Affiliation_Recall", 0.0),
        )
        or 0.0
    )
    if not np.isfinite(precision):
        precision = 0.0
    if not np.isfinite(recall):
        recall = 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    individual_precision = result.get(
        "individual_precision_probabilities",
        [precision],
    )
    individual_recall = result.get(
        "individual_recall_probabilities",
        [recall],
    )
    return {
        "f1": float(f1),
        "precision": precision,
        "recall": recall,
        "individual_precision_probabilities": [
            float(value) for value in individual_precision
        ],
        "individual_recall_probabilities": [
            float(value) for value in individual_recall
        ],
    }


def _affiliation_from_prediction(prediction, y):
    """Compute official affiliation precision, recall, and F1."""
    details = _affiliation_details_from_prediction(prediction, y)
    return details["f1"], details["precision"], details["recall"]


def compute_affiliation(score, y, *, threshold=None):
    """Compute exact official affiliation, optionally at one fixed threshold."""
    score = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    if score.shape != y.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if threshold is not None:
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.number))
            or not np.isfinite(float(threshold))
        ):
            raise ValueError("Affiliation threshold must be a finite number")
        return _affiliation_from_prediction(score >= float(threshold), y)

    thresholds = np.concatenate(
        ([np.nextafter(float(np.min(score)), -np.inf)], np.unique(score))
    )
    best = (0.0, 0.0, 0.0)
    for threshold in thresholds:
        candidate = _affiliation_from_prediction(
            (score > threshold).astype(np.int32),
            y,
        )
        if candidate[0] > best[0]:
            best = candidate
    return best


def _exact_point_metrics(scores, labels, point_adjusted=False, lengths=None):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if not np.isfinite(scores).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    candidates = np.concatenate(
        (
            [np.nextafter(float(np.max(scores)), np.inf)],
            np.unique(scores),
            [np.nextafter(float(np.min(scores)), -np.inf)],
        )
    )
    best = {
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": None,
    }
    for threshold in candidates:
        if point_adjusted:
            if lengths is None:
                raise ValueError("Segment lengths are required for point adjustment")
            score_parts = []
            label_parts = []
            offset = 0
            for length in lengths:
                score_parts.append(scores[offset:offset + length])
                label_parts.append(labels[offset:offset + length])
                offset += length
            predictions, actuals = point_adjust_segments(
                score_parts, label_parts, threshold
            )
            prediction = np.concatenate(predictions)
            actual = np.concatenate(actuals)
        else:
            prediction = scores >= threshold
            actual = labels.astype(bool, copy=False)
        tp = int(np.sum(prediction & actual))
        fp = int(np.sum(prediction & ~actual))
        fn = int(np.sum(~prediction & actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold),
            }
    return best


def _trapezoidal_area(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _trapezoidal_pr_auc(labels, scores):
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    return _trapezoidal_area(recall[::-1], precision[::-1])


def _exact_point_metrics(scores, labels, point_adjusted=False, lengths=None):
    """Compute exact point metrics with vectorized score-group sweeps."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("Scores and labels must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if point_adjusted:
        if lengths is None or sum(lengths) != len(scores):
            raise ValueError("Segment lengths must cover all scores")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    )
    ends = np.r_[starts[1:] - 1, len(scores) - 1]
    unique_desc = sorted_scores[ends]
    cumulative_tp = np.cumsum(labels[order].astype(np.int64))
    tp_desc = cumulative_tp[ends].astype(np.float64)
    predicted_desc = ends.astype(np.float64) + 1.0
    unique_asc = unique_desc[::-1]
    tp_asc = tp_desc[::-1]
    predicted_asc = predicted_desc[::-1]

    if point_adjusted:
        extra_diff = np.zeros(len(unique_asc) + 1, dtype=np.float64)
        offset = 0
        for length in lengths:
            segment_labels = labels[offset:offset + length].astype(bool)
            padded = np.r_[False, segment_labels, False]
            event_starts = np.flatnonzero(~padded[:-1] & padded[1:])
            event_ends = np.flatnonzero(padded[:-1] & ~padded[1:])
            for start, end in zip(event_starts, event_ends):
                event_scores = scores[offset + start:offset + end]
                event_indices = np.searchsorted(
                    unique_asc, event_scores, side="left"
                )
                max_index = int(event_indices.max())
                counts = np.bincount(
                    event_indices, minlength=len(unique_asc)
                )
                lower_indices = np.flatnonzero(counts[:max_index])
                if lower_indices.size:
                    lower_counts = counts[lower_indices]
                    extra_diff[lower_indices + 1] += lower_counts
                    extra_diff[max_index + 1] -= float(lower_counts.sum())
            offset += length
        extra_asc = np.cumsum(extra_diff[:-1])
        tp_asc = tp_asc + extra_asc

    fp_asc = predicted_asc - (
        cumulative_tp[ends][::-1].astype(np.float64)
    )
    positive_count = float(labels.sum())
    fn_asc = positive_count - tp_asc
    precision_asc = np.divide(
        tp_asc,
        tp_asc + fp_asc,
        out=np.zeros_like(tp_asc),
        where=(tp_asc + fp_asc) > 0,
    )
    recall_asc = np.divide(
        tp_asc,
        tp_asc + fn_asc,
        out=np.zeros_like(tp_asc),
        where=(tp_asc + fn_asc) > 0,
    )
    f1_asc = np.divide(
        2.0 * precision_asc * recall_asc,
        precision_asc + recall_asc,
        out=np.zeros_like(precision_asc),
        where=(precision_asc + recall_asc) > 0,
    )

    f1_candidates = np.r_[0.0, f1_asc, f1_asc[-1]]
    selected_index = int(np.argmax(f1_candidates))
    if selected_index == 0:
        threshold = np.nextafter(float(unique_desc[0]), np.inf)
    elif selected_index == len(unique_asc) + 1:
        threshold = np.nextafter(float(unique_asc[0]), -np.inf)
    else:
        threshold = float(unique_asc[selected_index - 1])
    return {
        "f1": float(f1_candidates[selected_index]),
        "precision": float(
            np.r_[0.0, precision_asc, precision_asc[-1]][selected_index]
        ),
        "recall": float(
            np.r_[0.0, recall_asc, recall_asc[-1]][selected_index]
        ),
        "threshold": threshold,
    }


def compute_segmented_metrics(
    segment_scores,
    segment_labels,
    range_window=30,
    vus_max_window=60,
):
    """Compute the formal metrics without crossing independent segments."""
    if not isinstance(segment_scores, Mapping) or not isinstance(
        segment_labels, Mapping
    ):
        raise TypeError("Segment scores and labels must be mappings")
    if list(segment_scores) != list(segment_labels):
        raise ValueError("Segment score and label IDs must match in order")
    if not segment_scores:
        raise ValueError("At least one segment is required")

    score_parts = []
    label_parts = []
    segment_details = {}
    lengths = []
    for segment_id in segment_scores:
        scores = np.asarray(segment_scores[segment_id], dtype=np.float64).ravel()
        labels = np.asarray(segment_labels[segment_id]).ravel()
        if scores.size == 0 or scores.shape != labels.shape:
            raise ValueError(f"Invalid score/label shape for segment {segment_id}")
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Non-finite score in segment {segment_id}")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Labels must be binary in segment {segment_id}")
        labels = labels.astype(np.int32, copy=False)
        score_parts.append(scores)
        label_parts.append(labels)
        lengths.append(len(scores))

        range_valid = np.unique(labels).size == 2
        range_metrics = dict.fromkeys(("R-AUC-ROC", "R-AUC-PR", "VUS-ROC", "VUS-PR"))
        if range_valid:
            range_roc, range_pr = RangeAUC_vectorized(
                labels, scores, window=int(range_window)
            )
            vus_roc, vus_pr = RangeAUC_volume(
                labels, scores, windowSize=int(vus_max_window)
            )
            range_metrics.update(zip(range_metrics, map(float, (
                range_roc, range_pr, vus_roc, vus_pr
            ))))
        segment_details[str(segment_id)] = {
            "length": len(scores),
            "weight": 0.0,
            "range_vus_valid": bool(range_valid),
            "range_vus_exclusion_reason": None if range_valid else "single_label_class",
            "metrics": range_metrics,
        }

    pooled_scores = np.concatenate(score_parts)
    pooled_labels = np.concatenate(label_parts)
    valid_range_length = sum(
        details["length"] for details in segment_details.values()
        if details["range_vus_valid"]
    )
    if valid_range_length == 0:
        raise ValueError(
            "Range/VUS metrics are undefined: no test segment contains both label classes; "
            "a complete 13-metric report cannot be produced"
        )
    for details in segment_details.values():
        details["weight"] = (
            details["length"] / valid_range_length if details["range_vus_valid"] else 0.0
        )

    point = _exact_point_metrics(pooled_scores, pooled_labels)
    point_adjusted = _exact_point_metrics(
        pooled_scores,
        pooled_labels,
        point_adjusted=True,
        lengths=lengths,
    )
    affiliation_threshold = point_adjusted["threshold"]
    affiliation_precisions = []
    affiliation_recalls = []
    for segment_id, scores, labels in zip(
        segment_details,
        score_parts,
        label_parts,
    ):
        affiliation = _affiliation_details_from_prediction(
            scores >= affiliation_threshold,
            labels,
        )
        segment_details[segment_id]["metrics"]["Aff-F1"] = affiliation["f1"]
        affiliation_precisions.extend(
            value
            for value in affiliation[
                "individual_precision_probabilities"
            ]
            if np.isfinite(value)
        )
        affiliation_recalls.extend(
            value
            for value in affiliation["individual_recall_probabilities"]
            if np.isfinite(value)
        )
    affiliation_precision = (
        float(np.mean(affiliation_precisions))
        if affiliation_precisions
        else 0.0
    )
    affiliation_recall = (
        float(np.mean(affiliation_recalls))
        if affiliation_recalls
        else 0.0
    )
    affiliation_f1 = (
        2.0
        * affiliation_precision
        * affiliation_recall
        / (affiliation_precision + affiliation_recall)
        if affiliation_precision + affiliation_recall
        else 0.0
    )
    auc_roc = roc_auc_score(pooled_labels, pooled_scores)
    auc_pr = _trapezoidal_pr_auc(pooled_labels, pooled_scores)
    weighted = {
        metric: sum(
            details["metrics"][metric] * details["weight"]
            for details in segment_details.values()
            if details["range_vus_valid"]
        )
        for metric in ("R-AUC-ROC", "R-AUC-PR", "VUS-ROC", "VUS-PR")
    }
    metrics = {
        "P": point["precision"],
        "R": point["recall"],
        "F1": point["f1"],
        "P_PA": point_adjusted["precision"],
        "R_PA": point_adjusted["recall"],
        "F1_PA": point_adjusted["f1"],
        "Aff-F1": float(affiliation_f1),
        **weighted,
        "AUC-ROC": float(auc_roc),
        "AUC-PR": float(auc_pr),
    }
    return {
        "metrics": {name: float(metrics[name]) for name in ALL_METRICS},
        "segments": segment_details,
        "thresholds": {
            "point": point["threshold"],
            "point_adjusted": point_adjusted["threshold"],
            "affiliation": affiliation_threshold,
        },
        "segment_aggregation": "metric_specific_no_cross_segment_boundaries",
        "affiliation_aggregation": (
            "ground_truth_event_macro_precision_recall_then_f1"
        ),
        "range_vus_aggregation": "valid_segment_length_weighted",
        "range_vus_single_class_policy": "exclude_undefined_segments_record_null",
    }


def build_evaluation_report(
    *,
    metrics,
    segment_details,
    dataset,
    strategy,
    post_processing,
    seed,
    imputation_rounds,
    thresholds,
    range_window,
    vus_max_window,
    architecture,
    checkpoint_identity,
    source_fingerprint,
    test_manifest,
):
    """Build the stable formal JSON envelope."""
    if set(metrics) != set(ALL_METRICS):
        raise ValueError("Report metrics must contain exactly the 13 required metrics")
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Report metrics must be finite")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Report dataset must be a non-empty string")
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in source_fingerprint)
    ):
        raise ValueError("Report source_fingerprint must be a lowercase SHA256")
    if not isinstance(test_manifest, Mapping) or not test_manifest:
        raise ValueError("Report test_manifest must be a non-empty mapping")
    return {
        "metrics": {name: float(metrics[name]) for name in ALL_METRICS},
        "segments": segment_details,
        "protocol": {
            "strategy": strategy,
            "post_processing": post_processing,
            "seed": int(seed),
            "imputation_rounds": int(imputation_rounds),
            "thresholds": thresholds,
            "metric_threshold_selection": "test_oracle",
            "segment_aggregation": (
                "metric_specific_no_cross_segment_boundaries"
            ),
            "affiliation_aggregation": (
                "ground_truth_event_macro_precision_recall_then_f1"
            ),
            "range_vus_aggregation": "valid_segment_length_weighted",
        "range_vus_single_class_policy": "exclude_undefined_segments_record_null",
            "affiliation_threshold_selection": (
                "global_point_adjusted_f1_oracle"
            ),
            "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
            "range_auc_implementation": "official_metricor.RangeAUC",
            "range_auc_window": int(range_window),
            "range_auc_thresholds": OFFICIAL_VUS_THRESHOLD_COUNT,
            "vus_implementation": "official_metricor.RangeAUC_volume_opt",
            "vus_version": "opt",
            "vus_max_window": int(vus_max_window),
            "vus_thresholds": OFFICIAL_VUS_THRESHOLD_COUNT,
            "vus_repository": OFFICIAL_VUS_REPOSITORY,
            "vus_commit": OFFICIAL_VUS_COMMIT,
            "architecture": architecture,
            "checkpoint_identity": checkpoint_identity,
            "dataset": dataset.strip().lower(),
            "source_fingerprint": source_fingerprint,
            "test_manifest": dict(test_manifest),
        },
    }


def compute_all_metrics(score, y):
    """Compute all 13 metrics for one segment using the formal protocol."""
    result = compute_segmented_metrics(
        {"segment": np.asarray(score)},
        {"segment": np.asarray(y)},
        range_window=30,
        vus_max_window=60,
    )
    return result["metrics"]


def _load_test_segments(
    loader_fn,
    data_root,
    *,
    protocol_manifest,
    window_size,
):
    loaded = loader_fn(data_root)
    test_segments = materialize_manifest_test_segments(
        loaded,
        protocol_manifest,
        window_size=window_size,
    )
    values = {}
    labels = {}
    for segment in test_segments:
        if segment.labels is None:
            raise ValueError(f"Test segment {segment.segment_id} has no labels")
        values[segment.segment_id] = np.asarray(segment.values)
        labels[segment.segment_id] = np.asarray(segment.labels)
    return values, labels


def _load_validation_segments(
    loader_fn,
    data_root,
    *,
    protocol_manifest,
    window_size,
):
    loaded = loader_fn(data_root)
    validation_segments = materialize_manifest_validation_segments(
        loaded,
        protocol_manifest,
        window_size=window_size,
    )
    return {
        segment.segment_id: np.asarray(segment.values)
        for segment in validation_segments
    }


def _validation_id_by_test_id(protocol_manifest):
    if not isinstance(protocol_manifest, Mapping):
        raise TypeError("Protocol manifest must be a mapping")
    if protocol_manifest.get("protocol_version") == PAPER_PROTOCOL_VERSION:
        return {record["test_segment_id"]: record["validation_segment_id"]
                for record in protocol_manifest["sources"].values()}
    if protocol_manifest.get("protocol_version") == OFFICIAL_PROTOCOL_VERSION:
        test_segments = protocol_manifest.get("test_segments")
        if not isinstance(test_segments, Mapping):
            raise ValueError(
                "Official protocol manifest is missing test segment records"
            )
        references = {}
        for source_id, record in test_segments.items():
            if not isinstance(source_id, str) or not isinstance(record, Mapping):
                raise ValueError(
                    "Official protocol test segment records are invalid"
                )
            test_id = record.get("test_segment_id")
            validation_id = record.get("validation_segment_id")
            if (
                not isinstance(test_id, str)
                or test_id in references
                or (
                    validation_id is not None
                    and not isinstance(validation_id, str)
                )
            ):
                raise ValueError(
                    "Official protocol test/validation segment IDs are invalid"
                )
            if validation_id is not None:
                references[test_id] = validation_id
        return references
    entities = protocol_manifest.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("Protocol manifest is missing entity records")
    references = {}
    for entity_id, record in entities.items():
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Protocol manifest entity {entity_id!r} must be a mapping"
            )
        validation_id = record.get("validation_segment_id")
        test_ids = record.get("test_segment_ids")
        if not isinstance(validation_id, str) or not isinstance(
            test_ids, list
        ):
            raise ValueError(
                f"Protocol manifest entity {entity_id!r} has invalid segment IDs"
            )
        for test_id in test_ids:
            if not isinstance(test_id, str) or test_id in references:
                raise ValueError("Protocol manifest test IDs are invalid")
            references[test_id] = validation_id
    return references


def _score_segments(
    model,
    values,
    scaler_state,
    device,
    n_rounds,
    seed,
    *,
    include_components=False,
    include_uncertainty=False,
):
    if include_uncertainty and not include_components:
        raise ValueError(
            "Masked-imputation uncertainty requires component scoring"
        )
    scored = {}
    components = {} if include_components else None
    for segment_id, raw_values in values.items():
        standardized = standardize_test_data(raw_values, scaler_state)
        if include_components:
            if include_uncertainty:
                segment_components = (
                    compute_imp_score_components_with_uncertainty(
                        model,
                        standardized,
                        W=model.time_len,
                        n_rounds=n_rounds,
                        device=device,
                        seed=seed,
                    )
                )
            else:
                segment_components = compute_imp_score_components(
                    model,
                    standardized,
                    W=model.time_len,
                    n_rounds=n_rounds,
                    device=device,
                    seed=seed,
                )
            components[segment_id] = segment_components
            scored[segment_id] = segment_components["anom_score"]
        else:
            scored[segment_id] = compute_imp_score(
                model,
                standardized,
                W=model.time_len,
                n_rounds=n_rounds,
                device=device,
                seed=seed,
            )
    return scored, components


def evaluate_single(model, loader_fn, data_root, device='cuda', strategy='mv_sr_rank',
                    fuse_alpha=0.3, dilation=0, n_rounds=3, scaler_state=None,
                    seed=42, protocol_manifest=None):
    """Evaluate each test segment independently with a fixed strategy."""
    values, labels = _load_test_segments(
        loader_fn,
        data_root,
        protocol_manifest=protocol_manifest,
        window_size=model.time_len,
    )
    imp_scores, component_scores = _score_segments(
        model,
        values,
        scaler_state,
        device,
        n_rounds,
        seed,
        include_components=strategy in {
            "mv_rank",
            "mv_sr_rank",
            "mv_sr_unc_rank",
            "mv_sr_valcal",
            "mv_sr_trimcal",
        },
        include_uncertainty=strategy == "mv_sr_unc_rank",
    )
    sr_scores = None
    if strategy in {
        'sr',
        'fuse',
        'ms_sr',
        'ms_fuse',
        'mv_sr_rank',
        'mv_sr_unc_rank',
        'mv_sr_valcal',
        'mv_sr_trimcal',
    }:
        sr_scores = dict(zip(
            values,
            compute_sr_score_segments(list(values.values())),
        ))
    validation_components = None
    validation_sr_scores = None
    validation_reference_ids = None
    uses_validation_calibration = strategy in {
        "mv_sr_valcal",
        "mv_sr_trimcal",
    }
    if uses_validation_calibration:
        validation_values = _load_validation_segments(
            loader_fn,
            data_root,
            protocol_manifest=protocol_manifest,
            window_size=model.time_len,
        )
        _validation_imp, validation_components = _score_segments(
            model,
            validation_values,
            scaler_state,
            device,
            n_rounds,
            seed,
            include_components=True,
        )
        validation_sr_scores = dict(zip(
            validation_values,
            compute_sr_score_segments(list(validation_values.values())),
        ))
        validation_reference_ids = _validation_id_by_test_id(
            protocol_manifest
        )
    selected = {}
    for segment_id, imp_raw in imp_scores.items():
        validation_id = (
            None
            if validation_reference_ids is None
            else validation_reference_ids.get(segment_id)
        )
        if uses_validation_calibration and (
            validation_id is None
            or validation_components is None
            or validation_sr_scores is None
            or validation_id not in validation_components
            or validation_id not in validation_sr_scores
        ):
            raise ValueError(
                f"Test segment {segment_id!r} has no validation reference"
            )
        selected[segment_id] = select_strategy(
            imp_raw,
            None if sr_scores is None else sr_scores[segment_id],
            strategy=strategy,
            fuse_alpha=fuse_alpha,
            dilation=dilation,
            component_scores=(
                None
                if component_scores is None
                else component_scores[segment_id]
            ),
            validation_components=(
                None
                if validation_id is None or validation_components is None
                else validation_components[validation_id]
            ),
            validation_spectral_residual=(
                None
                if validation_id is None or validation_sr_scores is None
                else validation_sr_scores[validation_id]
            ),
        )
    report = compute_segmented_metrics(
        selected, labels, range_window=30, vus_max_window=60
    )
    return strategy.upper(), report


def evaluate_ucr_chunked(model, loader_fn, data_root, device='cuda',
                         chunk_size=500000, strategy='mv_sr_rank', fuse_alpha=0.3,
                         dilation=0, n_rounds=3, scaler_state=None, seed=42,
                         protocol_manifest=None):
    """Evaluate UCR segment by segment to preserve real series boundaries."""
    return evaluate_single(
        model, loader_fn, data_root, device=device, strategy=strategy,
        fuse_alpha=fuse_alpha, dilation=dilation, n_rounds=n_rounds,
        scaler_state=scaler_state, seed=seed,
        protocol_manifest=protocol_manifest,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified 13-metric evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=sorted(DATASET_REGISTRY), help="Dataset name")
    parser.add_argument("--data_root", type=str, default="datasets",
                        help="Root directory for datasets")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--strategy",
        choices=[
            "imp",
            "sr",
            "fuse",
            "ms_imp",
            "ms_sr",
            "ms_fuse",
            "mv_rank",
            "mv_sr_rank",
            "mv_sr_unc_rank",
            "mv_sr_valcal",
            "mv_sr_trimcal",
        ],
        default="mv_sr_rank", help="Fixed scoring strategy; never selected using test labels",
    )
    parser.add_argument("--fuse_alpha", type=float, default=0.3,
                        help="Spectral-residual weight for fixed fusion strategies")
    parser.add_argument("--dilation", type=int, default=0,
                        help="Optional fixed maximum-filter width (0 disables it)")
    parser.add_argument("--n_rounds", type=int, default=3,
                        help="Number of stochastic imputation rounds")
    args = parser.parse_args()

    if args.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {args.dataset}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    loader_fn = DATASET_REGISTRY[args.dataset]
    data_root = os.path.join(
        args.data_root, DATASET_DIRECTORY_NAMES[args.dataset]
    )
    print(f"Checkpoint: {args.checkpoint}")

    model, checkpoint = load_model_checkpoint(args.checkpoint, device=args.device)
    scaler_state = checkpoint.get("scaler_state")
    _scaler_parameters(scaler_state, model.num_vars)
    try:
        protocol_manifest = checkpoint["resolved_config"]["data_protocol"][
            "manifest"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Checkpoint is missing the unified entity-split manifest"
        ) from exc
    if not 0.0 <= args.fuse_alpha <= 1.0:
        raise ValueError("--fuse_alpha must be in [0, 1]")
    if args.dilation < 0:
        raise ValueError("--dilation cannot be negative")
    print(
        f"Dataset: {args.dataset}  Model: M2B-MoE  Strategy: {args.strategy}"
    )

    # Evaluate
    if args.dataset == "ucr":
        label, evaluation = evaluate_ucr_chunked(
            model, loader_fn, data_root, device=args.device,
            strategy=args.strategy, fuse_alpha=args.fuse_alpha,
            dilation=args.dilation, n_rounds=args.n_rounds,
            scaler_state=scaler_state,
            protocol_manifest=protocol_manifest,
        )
    else:
        label, evaluation = evaluate_single(
            model, loader_fn, data_root, device=args.device,
            strategy=args.strategy, fuse_alpha=args.fuse_alpha,
            dilation=args.dilation, n_rounds=args.n_rounds,
            scaler_state=scaler_state,
            protocol_manifest=protocol_manifest,
        )
    metrics = evaluation["metrics"]

    # Print results
    print(f"\nFixed strategy: {label}")
    print("=" * 70)
    print(f"{'Metric':<15} {'Value':>10}  {'>0.65?':>8}")
    print("-" * 35)
    goal_count = 0
    for k in ALL_METRICS:
        v = metrics.get(k, 0)
        mark = "  *" if v > 0.65 else ""
        if v > 0.65: goal_count += 1
        print(f"{k:<15} {v:>10.4f}{mark}")
    print("-" * 35)
    print(f"Goal: {goal_count}/{len(ALL_METRICS)} metrics > 0.65")

    if args.output:
        digest = hashlib.sha256()
        with open(args.checkpoint, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        report = build_evaluation_report(
            metrics=metrics,
            segment_details=evaluation["segments"],
            dataset=args.dataset,
            strategy=args.strategy,
            post_processing={
                "fuse_alpha": args.fuse_alpha,
                "dilation": args.dilation,
                "multi_view_rank_weights": dict(_MULTI_VIEW_RANK_WEIGHTS),
                "multi_view_spectral_rank_weights": dict(
                    _MULTI_VIEW_SPECTRAL_RANK_WEIGHTS
                ),
                "multi_view_spectral_uncertainty_rank_weights": dict(
                    _MULTI_VIEW_SPECTRAL_UNCERTAINTY_RANK_WEIGHTS
                ),
                "validation_calibration": (
                    "median_mad_asinh_entity_local"
                    if args.strategy == "mv_sr_valcal"
                    else (
                        "lower_80_percent_median_mad_asinh_entity_local"
                        if args.strategy == "mv_sr_trimcal"
                        else "none"
                    )
                ),
            },
            seed=int(checkpoint.get("seed", 42)),
            imputation_rounds=args.n_rounds,
            thresholds=evaluation["thresholds"],
            range_window=30,
            vus_max_window=60,
            architecture=checkpoint["model_config"]["architecture"],
            checkpoint_identity=f"sha256:{digest.hexdigest()}",
            source_fingerprint=checkpoint["source_fingerprint"],
            test_manifest=protocol_manifest.get(
                "test_segments",
                protocol_manifest.get("sources", protocol_manifest.get("entities", {})),
            ),
        )
        report["label"] = label
        report["goal"] = goal_count
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
