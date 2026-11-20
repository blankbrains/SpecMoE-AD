import numpy as np
import math
from typing import List, Tuple, Dict, Optional


# ==========================================
# Part 1: 基础工具函数 (Numpy 极速版)
# ==========================================

def convert_vector_to_events(vector: np.ndarray) -> List[Tuple[int, int]]:
    """
    将二进制向量转换为事件列表。
    返回格式: [(start, end), ... ] (exclusive end)
    """
    if len(vector) == 0:
        return []

    vector = np.asarray(vector)
    padded = np.concatenate([[0], vector, [0]])
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # 确保返回列表而非 zip 迭代器
    return list(zip(starts.tolist(), ends.tolist()))


def range_convers_new(label: np.ndarray) -> List[Tuple[int, int]]:
    """
    将二进制标签转换为 inclusive 的 ``(start, end)`` 区间。
    """
    label = np.asarray(label)
    if len(label) == 0:
        return []

    label_binary = (label != 0).astype(np.int8)
    padded = np.concatenate([[0], label_binary, [0]])
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    # inclusive end
    ends = np.where(diffs == -1)[0] - 1

    return list(zip(starts.tolist(), ends.tolist()))


def interval_intersection(I: Optional[Tuple], J: Optional[Tuple]) -> Optional[Tuple]:
    if I is None or J is None: return None
    # 展开 max/min 减少函数调用开销
    start = I[0] if I[0] > J[0] else J[0]
    end = I[1] if I[1] < J[1] else J[1]
    if start >= end: return None
    return (start, end)


def interval_length(I: Optional[Tuple]) -> float:
    return I[1] - I[0] if I is not None else 0


def sum_interval_lengths(intervals: List) -> float:
    return sum(interval_length(I) for I in intervals if I is not None)


# ==========================================
# Part 2, 3, 4: Affiliation Metrics
# (逻辑复杂且非主要瓶颈，保持逻辑精确性)
# ==========================================

def t_start(j: int, Js: List[Tuple], Trange: Tuple) -> float:
    if j == len(Js):
        return 2 * max(Trange) - t_stop(len(Js) - 1, Js, Trange)
    return Js[j][0]


def t_stop(j: int, Js: List[Tuple], Trange: Tuple) -> float:
    if j == -1:
        return 2 * min(Trange) - t_start(0, Js, Trange)
    return Js[j][1]


def E_gt_func(j: int, Js: List[Tuple], Trange: Tuple) -> Tuple[float, float]:
    return ((t_stop(j - 1, Js, Trange) + t_start(j, Js, Trange)) / 2,
            (t_stop(j, Js, Trange) + t_start(j + 1, Js, Trange)) / 2)


def get_all_E_gt_func(Js: List[Tuple], Trange: Tuple) -> List[Tuple[float, float]]:
    return [E_gt_func(j, Js, Trange) for j in range(len(Js))]


def affiliation_partition(Is: List[Tuple], E_gt: List[Tuple]) -> List[List]:
    out = []
    # 简单空间换时间优化：避免重复计算
    for E_gt_j in E_gt:
        zone_intervals = []
        for I in Is:
            # 快速过滤
            if I[1] <= E_gt_j[0] or I[0] >= E_gt_j[1]: continue

            # 展开 intersection 逻辑
            start = I[0] if I[0] > E_gt_j[0] else E_gt_j[0]
            end = I[1] if I[1] < E_gt_j[1] else E_gt_j[1]
            if start < end:
                zone_intervals.append((start, end))
        out.append(zone_intervals)
    return out


# ==========================================
# Part 3: Affiliation 积分计算 (保持原样)
# ==========================================

def cut_into_three_func(I: Optional[Tuple], J: Tuple) -> Tuple:
    if I is None: return (None, None, None)
    I_inter_J = interval_intersection(I, J)
    if I_inter_J is None:
        if I[1] <= J[0]:
            return (I, None, None)
        elif I[0] >= J[1]:
            return (None, None, I)
        else:
            return (None, None, None)
    I_before = (I[0], J[0]) if I[0] < J[0] else None
    I_after = (J[1], I[1]) if I[1] > J[1] else None
    return (I_before, I_inter_J, I_after)


def integral_mini_interval_P_CDFmethod__min_piece(I: Tuple, J: Tuple, E: Tuple) -> float:
    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    C = 0.5 * A + m * B
    return C


def integral_mini_interval_Pprecision_CDFmethod(I: Tuple, J: Tuple, E: Tuple) -> float:
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)
    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = 0.5 * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)
    DeltaI = i_max - i_min
    DeltaE = e_max - e_min
    if DeltaE == 0: return 0
    return DeltaI - (1 / DeltaE) * (integral_min_piece + integral_linear_piece + integral_remaining_piece)


def integral_interval_probaCDF_precision(I: Optional[Tuple], J: Tuple, E: Tuple) -> float:
    if I is None: return 0
    cut = cut_into_three_func(I, J)
    d_left = integral_mini_interval_Pprecision_CDFmethod(cut[0], J, E) if cut[0] else 0
    d_middle = cut[1][1] - cut[1][0] if cut[1] else 0
    d_right = integral_mini_interval_Pprecision_CDFmethod(cut[2], J, E) if cut[2] else 0
    return d_left + d_middle + d_right


def get_pivot_j(I: Tuple, J: Tuple) -> float:
    if I[1] <= J[0]:
        return J[0]
    elif I[0] >= J[1]:
        return J[1]
    else:
        raise ValueError("I and J should not intersect")


def cut_J_based_on_mean_func(J: Optional[Tuple], e_mean: float) -> Tuple:
    if J is None: return (None, None)
    if e_mean >= J[1]:
        return (J, None)
    elif e_mean <= J[0]:
        return (None, J)
    else:
        return ((J[0], e_mean), (e_mean, J[1]))


def integral_mini_interval_Precall_CDFmethod(I: Tuple, J: Tuple, E: Tuple) -> float:
    try:
        i_pivot = get_pivot_j(J, I)
    except ValueError:
        return 0
    e_min, e_max = E[0], E[1]
    if i_pivot <= e_min or i_pivot >= e_max: return 0
    e_mean = (e_min + e_max) / 2
    J_before, J_after = cut_J_based_on_mean_func(J, e_mean)
    iemin_mean = (e_min + i_pivot) / 2
    J_before_closeE, J_before_closeI = cut_J_based_on_mean_func(J_before, iemin_mean)
    iemax_mean = (e_max + i_pivot) / 2
    J_after_closeI, J_after_closeE = cut_J_based_on_mean_func(J_after, iemax_mean)

    def get_bounds(interval):
        return (interval[0], interval[1]) if interval else (math.nan, math.nan)

    j_bb_min, j_bb_max = get_bounds(J_before_closeE)
    j_ba_min, j_ba_max = get_bounds(J_before_closeI)
    j_ab_min, j_ab_max = get_bounds(J_after_closeI)
    j_aa_min, j_aa_max = get_bounds(J_after_closeE)

    def safe_val(expr):
        return 0 if math.isnan(expr) else expr

    if i_pivot >= J[1]:
        p1 = safe_val((i_pivot - e_min) * (j_bb_max - j_bb_min))
        p2 = safe_val(2 * i_pivot * (j_ba_max - j_ba_min) - (j_ba_max ** 2 - j_ba_min ** 2))
        p3 = safe_val(2 * i_pivot * (j_ab_max - j_ab_min) - (j_ab_max ** 2 - j_ab_min ** 2))
        p4 = safe_val((e_max + i_pivot) * (j_aa_max - j_aa_min) - (j_aa_max ** 2 - j_aa_min ** 2))
    else:
        p1 = safe_val((j_bb_max ** 2 - j_bb_min ** 2) - (e_min + i_pivot) * (j_bb_max - j_bb_min))
        p2 = safe_val((j_ba_max ** 2 - j_ba_min ** 2) - 2 * i_pivot * (j_ba_max - j_ba_min))
        p3 = safe_val((j_ab_max ** 2 - j_ab_min ** 2) - 2 * i_pivot * (j_ab_max - j_ab_min))
        p4 = safe_val((e_max - i_pivot) * (j_aa_max - j_aa_min))

    out_integral = p1 + p2 + p3 + p4
    DeltaJ = J[1] - J[0]
    DeltaE = e_max - e_min
    return DeltaJ - (1 / DeltaE) * out_integral if DeltaE > 0 else 0


def integral_interval_probaCDF_recall(I: Optional[Tuple], J: Optional[Tuple], E: Tuple) -> float:
    if I is None or J is None: return 0
    cut = cut_into_three_func(J, I)
    d_left = integral_mini_interval_Precall_CDFmethod(I, cut[0], E) if cut[0] else 0
    d_middle = cut[1][1] - cut[1][0] if cut[1] else 0
    d_right = integral_mini_interval_Precall_CDFmethod(I, cut[2], E) if cut[2] else 0
    return d_left + d_middle + d_right


def affiliation_precision_proba(Is: List, J: Tuple, E: Tuple) -> float:
    Is_filtered = [I for I in Is if I is not None]
    if not Is_filtered: return math.nan
    total_length = sum_interval_lengths(Is_filtered)
    if total_length == 0: return math.nan
    total_integral = sum(integral_interval_probaCDF_precision(I, J, E) for I in Is_filtered)
    return total_integral / total_length


def affiliation_recall_proba(Is: List, J: Tuple, E: Tuple) -> float:
    Is_filtered = [I for I in Is if I is not None]
    if not Is_filtered: return 0
    j_length = interval_length(J)
    if j_length == 0: return 0
    E_gt_recall = get_all_E_gt_func(Is_filtered, E)
    Js_partitioned = affiliation_partition([J], E_gt_recall)
    total_integral = 0
    for i, I in enumerate(Is_filtered):
        for J_part in Js_partitioned[i]:
            if J_part is not None:
                total_integral += integral_interval_probaCDF_recall(I, J_part, E)
    return total_integral / j_length


def pr_from_events(events_pred: List, events_gt: List, Trange: Tuple) -> Dict:
    if not events_gt:
        return {"Affiliation_Precision": math.nan, "Affiliation_Recall": math.nan}
    E_gt = get_all_E_gt_func(events_gt, Trange)
    Is_partitioned = affiliation_partition(events_pred, E_gt)
    precisions = []
    recalls = []
    for j, J in enumerate(events_gt):
        prec = affiliation_precision_proba(Is_partitioned[j], events_gt[j], E_gt[j])
        rec = affiliation_recall_proba(Is_partitioned[j], events_gt[j], E_gt[j])
        precisions.append(prec)
        recalls.append(rec)
    valid_precs = [p for p in precisions if not math.isnan(p)]
    valid_recs = [r for r in recalls if not math.isnan(r)]
    avg_precision = sum(valid_precs) / len(valid_precs) if valid_precs else 0
    avg_recall = sum(valid_recs) / len(valid_recs) if valid_recs else 0
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "individual_precision_probabilities": precisions,
        "individual_recall_probabilities": recalls,
        # Backward-compatible aliases used by the existing evaluator.
        "Affiliation_Precision": avg_precision,
        "Affiliation_Recall": avg_recall,
    }


# ==========================================
# Part 5: Range-AUC and VUS
# ==========================================

def extend_positive_range(labels: np.ndarray, window: int) -> np.ndarray:
    """Apply the soft label extension used by Range-AUC."""
    if window <= 0:
        return labels.astype(np.float64)

    # 复制并转换类型
    label = labels.astype(np.float64)
    length = len(label)

    L = range_convers_new(labels)
    if not L:
        return label

    # 针对每个 segment 进行处理
    # 这里的 Python 循环无法完全避免（每个 segment 长度不同，无法 tensor 化），
    # 但因为 segment 数量 << 数据点数量，这已经是性能瓶颈的最小化
    for s, e in L:
        # Forward extension (从 e 开始向后)
        end_f = min(e + window // 2, length)
        if end_f > e:
            x = np.arange(e, end_f)
            # 向量化计算 decay 值
            vals = np.sqrt(1 - (x - e) / window)
            # 原地更新最大值
            np.maximum(label[e:end_f], vals, out=label[e:end_f])

        # Backward extension (从 s 向前)
        start_b = max(s - window // 2, 0)
        if start_b < s:
            x = np.arange(start_b, s)
            vals = np.sqrt(1 - (s - x) / window)
            np.maximum(label[start_b:s], vals, out=label[start_b:s])

    # 裁剪超过 1.0 的值 (理论上不需要，但为了数值稳定)
    if label.max() > 1.0:
        np.minimum(label, 1.0, out=label)

    return label


def RangeAUC_vectorized(labels_original: np.ndarray, score: np.ndarray, window: int) -> Tuple[float, float]:
    """Compute range-aware ROC-AUC and PR-AUC."""
    score = np.asarray(score, dtype=np.float64)
    labels_extended = extend_positive_range(labels_original, window)

    P = float(np.sum(labels_original))
    P_new = (P + np.sum(labels_extended)) / 2

    L = range_convers_new(labels_extended)
    if not L:
        return 0.0, 0.0

    n = len(score)

    # 1. 准备阈值索引
    score_sorted_indices = np.argsort(score)[::-1]
    # 只需要值，不需要全排序后的数组
    score_sorted = score[score_sorted_indices]

    # 250 个采样点
    n_points = min(250, n)
    indices = np.linspace(0, n - 1, n_points).astype(np.int64)
    thresholds = score_sorted[indices]  # Shape: (K,)

    # 2. 计算 Recall (TP) - 基于累积和
    # 根据分数排序后的 labels
    labels_sorted = labels_extended[score_sorted_indices]
    cum_tp = np.cumsum(labels_sorted)

    # 通过索引直接获取 250 个点的 TP
    TPs = cum_tp[indices]

    # 计算 Recall 向量
    recalls = np.minimum(TPs / P_new, 1.0) if P_new > 0 else np.zeros_like(TPs)

    # 3. 计算 Existence (TPR_RangeAUC) - 矩阵比较
    # 预计算 segment max (数学等价于 sum > 0)
    segment_max_scores = np.array([np.max(score[s:e + 1]) for s, e in L])

    # Broadcasting: (M, 1) >= (1, K) -> (M, K)
    # 这一步将 N 次循环比较降维成一次矩阵操作
    existence_matrix = segment_max_scores[:, None] >= thresholds[None, :]
    existence_counts = np.sum(existence_matrix, axis=0)

    num_segments = len(L)
    existence_ratios = existence_counts / num_segments

    # 最终 TPR 向量
    TPRs = recalls * existence_ratios

    # 4. 计算 FPR & Precision - 向量化
    # total predicted positive = rank + 1
    total_predicted = indices + 1
    FPs = total_predicted - TPs

    N_new = n - P_new
    FPRs = FPs / N_new if N_new > 0 else np.zeros_like(FPs)

    Precisions = TPs / total_predicted
    # 修复除0 (rank 0 时 count 1, 不会除0，稳健起见)
    Precisions = np.nan_to_num(Precisions, nan=1.0)

    # 5. AUC 积分
    # 构造边界点 (与原代码逻辑一致)
    tpr_roc = np.concatenate(([0.0], TPRs, [1.0]))
    fpr_roc = np.concatenate(([0.0], FPRs, [1.0]))

    # AUC ROC
    width = fpr_roc[1:] - fpr_roc[:-1]
    height = (tpr_roc[1:] + tpr_roc[:-1]) / 2
    AUC_ROC = float(np.sum(width * height))

    # PR 曲线边界 (原代码逻辑)
    tpr_pr = np.concatenate(([0.0], TPRs, [1.0]))
    prec_pr = np.concatenate(([1.0], Precisions, [1.0]))  # 尾部补1.0

    # AUC PR
    width_PR = tpr_pr[1:] - tpr_pr[:-1]
    height_PR = (prec_pr[1:] + prec_pr[:-1]) / 2
    AUC_PR = float(np.sum(width_PR * height_PR))

    return AUC_ROC, AUC_PR


def RangeAUC_volume(labels_original: np.ndarray, score: np.ndarray,
                    windowSize: int) -> Tuple[float, float]:
    """Compute VUS by averaging Range-AUC over every window size."""
    auc_rocs = []
    auc_prs = []

    for w in range(windowSize + 1):
        auc_roc, auc_pr = RangeAUC_vectorized(labels_original, score, w)
        auc_rocs.append(auc_roc)
        auc_prs.append(auc_pr)

    n = len(auc_rocs)
    return sum(auc_rocs) / n, sum(auc_prs) / n


# ==========================================
# Part 6: 入口
# ==========================================

def evaluate_metrics(y_score: np.ndarray, y_true: np.ndarray,
                     slidingWindow: Optional[int] = None) -> Dict[str, float]:
    """Compute the four range-aware metrics used by the 13-metric protocol."""
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    if y_score.size == 0:
        raise ValueError("Metric inputs cannot be empty")
    if y_score.shape != y_true.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isfinite(y_score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")

    if slidingWindow is None:
        events = convert_vector_to_events(y_true)
        lengths = [e[1] - e[0] for e in events]
        slidingWindow = int(np.mean(lengths)) if lengths else 100
    if slidingWindow < 0:
        raise ValueError("slidingWindow cannot be negative")

    # Range-AUC
    r_auc_roc, r_auc_pr = RangeAUC_vectorized(y_true, y_score, slidingWindow)

    # VUS
    vus_roc, vus_pr = RangeAUC_volume(y_true, y_score, 2 * slidingWindow)

    return {
        "r_auc_roc": r_auc_roc,
        "r_auc_pr": r_auc_pr,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr
    }


def _trapezoidal_area(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _range_curve(labels_original: np.ndarray, score: np.ndarray,
                 window: int) -> Tuple[float, float]:
    """Compute a tie-safe range curve with bounded threshold sampling."""
    labels_original = np.asarray(labels_original).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if labels_original.shape != score.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isin(labels_original, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")

    extended = extend_positive_range(labels_original, window)
    total_positive = float(extended.sum())
    total_negative = float(len(score) - total_positive)
    ranges = range_convers_new(labels_original)

    order = np.argsort(score, kind="mergesort")[::-1]
    sorted_scores = score[order]
    sorted_labels = extended[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    )
    ends = np.r_[starts[1:] - 1, len(score) - 1]
    group_count = len(starts)
    sample_count = min(250, group_count)
    sampled_groups = np.unique(
        np.linspace(0, group_count - 1, sample_count).astype(np.int64)
    )
    selected_ends = ends[sampled_groups]
    thresholds = sorted_scores[selected_ends]
    cumulative_tp = np.cumsum(sorted_labels)
    tp = cumulative_tp[selected_ends]
    predicted = selected_ends.astype(np.float64) + 1.0
    fp = predicted - tp
    recall = tp / total_positive if total_positive else np.zeros_like(tp)
    if ranges:
        event_max = np.asarray(
            [score[start:end + 1].max() for start, end in ranges],
            dtype=np.float64,
        )
        detected = (event_max[:, None] >= thresholds[None, :]).sum(axis=0)
        recall *= detected / float(len(ranges))
    fpr = fp / total_negative if total_negative else np.zeros_like(fp)
    precision = np.divide(
        tp,
        predicted,
        out=np.ones_like(tp),
        where=predicted > 0,
    )

    roc = np.column_stack(
        (
            np.r_[0.0, fpr, 1.0],
            np.r_[0.0, recall, 1.0],
        )
    )
    roc = roc[np.argsort(roc[:, 0], kind="stable")]
    pr = np.column_stack(
        (
            np.r_[0.0, recall, 1.0],
            np.r_[1.0, precision, 1.0],
        )
    )
    pr = pr[np.argsort(pr[:, 0], kind="stable")]
    return (
        _trapezoidal_area(roc[:, 0], roc[:, 1]),
        _trapezoidal_area(pr[:, 0], pr[:, 1]),
    )


def RangeAUC_vectorized(labels_original: np.ndarray, score: np.ndarray,
                        window: int) -> Tuple[float, float]:
    """Compute tie-safe, bounded range ROC-AUC and PR-AUC."""
    return _range_curve(labels_original, score, window)


def RangeAUC_volume(labels_original: np.ndarray, score: np.ndarray,
                    windowSize: int) -> Tuple[float, float]:
    """Average range AUC over a bounded set of window sizes."""
    if windowSize < 0:
        raise ValueError("windowSize cannot be negative")
    if windowSize > 10:
        windows = np.unique(
            np.linspace(0, windowSize, 10).astype(np.int64)
        )
    else:
        windows = np.arange(windowSize + 1, dtype=np.int64)
    values = np.asarray(
        [RangeAUC_vectorized(labels_original, score, int(window))
         for window in windows],
        dtype=np.float64,
    )
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def evaluate_metrics(y_score: np.ndarray, y_true: np.ndarray,
                     slidingWindow: Optional[int] = None) -> Dict[str, float]:
    """Compute the four range-aware metrics used by the formal report."""
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).ravel()
    if y_score.size == 0:
        raise ValueError("Metric inputs cannot be empty")
    if y_score.shape != y_true.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isfinite(y_score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if slidingWindow is None:
        events = convert_vector_to_events(y_true)
        lengths = [end - start for start, end in events]
        slidingWindow = int(np.mean(lengths)) if lengths else 100
    r_auc_roc, r_auc_pr = RangeAUC_vectorized(
        y_true, y_score, int(slidingWindow)
    )
    vus_roc, vus_pr = RangeAUC_volume(
        y_true, y_score, int(2 * slidingWindow)
    )
    return {
        "r_auc_roc": r_auc_roc,
        "r_auc_pr": r_auc_pr,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
    }


# The definitions below reproduce the pinned official VUS implementation
# (TheDatumOrg/VUS commit 2ddf83b88e26118ae28674a2c422750733eb0f88).
# They intentionally override the historical local variants above.
OFFICIAL_VUS_REPOSITORY = "https://github.com/TheDatumOrg/VUS.git"
OFFICIAL_VUS_COMMIT = "2ddf83b88e26118ae28674a2c422750733eb0f88"
OFFICIAL_VUS_THRESHOLD_COUNT = 250
METRIC_PROTOCOL_REVISION = "official_vus_2ddf83b_aff_event_macro_v1"


def _validate_range_inputs(
    labels_original: np.ndarray,
    score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels_original).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if labels.size == 0 or labels.shape != score.shape:
        raise ValueError("Scores and labels must be non-empty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if not np.isfinite(score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("Range metrics require both label classes")
    return labels.astype(np.int32, copy=False), score


def extend_positive_range(labels: np.ndarray, window: int) -> np.ndarray:
    """Apply the official additive square-root range extension."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if window < 0:
        raise ValueError("window cannot be negative")

    extended = labels.astype(np.float64, copy=True)
    if window == 0:
        return extended
    half_window = window // 2
    for start, end_inclusive in range_convers_new(labels):
        forward = np.arange(
            end_inclusive + 1,
            min(end_inclusive + half_window + 1, len(labels)),
        )
        if forward.size:
            extended[forward] += np.sqrt(
                1.0 - (forward - end_inclusive) / float(window)
            )
        backward = np.arange(max(start - half_window, 0), start)
        if backward.size:
            extended[backward] += np.sqrt(
                1.0 - (start - backward) / float(window)
            )
    return np.minimum(np.ones(len(labels), dtype=np.float64), extended)


def _official_thresholds(score: np.ndarray) -> np.ndarray:
    score_sorted = -np.sort(-score)
    indices = np.linspace(
        0,
        len(score) - 1,
        OFFICIAL_VUS_THRESHOLD_COUNT,
    ).astype(np.int64)
    return score_sorted[indices]


def _merged_extended_sequences(
    sequence_original: List[Tuple[int, int]],
    length: int,
    window: int,
) -> List[Tuple[int, int]]:
    half_window = window // 2
    start = max(sequence_original[0][0] - half_window, 0)
    sequences = []
    for current, following in zip(
        sequence_original[:-1],
        sequence_original[1:],
    ):
        if current[1] + half_window < following[0] - half_window:
            sequences.append((start, current[1] + half_window))
            start = following[0] - half_window
    sequences.append(
        (
            start,
            min(sequence_original[-1][1] + half_window, length - 1),
        )
    )
    return sequences


def _range_curve(labels_original: np.ndarray, score: np.ndarray,
                 window: int) -> Tuple[float, float]:
    """Reproduce official ``metricor.RangeAUC`` with 250 thresholds."""
    if window < 0:
        raise ValueError("window cannot be negative")
    labels_original, score = _validate_range_inputs(labels_original, score)
    original_positive = float(labels_original.sum())
    extended = extend_positive_range(labels_original, window)
    ranges = range_convers_new(extended)
    thresholds = _official_thresholds(score)
    curve = np.zeros((OFFICIAL_VUS_THRESHOLD_COUNT + 2, 2))
    precisions = np.ones(OFFICIAL_VUS_THRESHOLD_COUNT + 1)
    positive_normalizer = (original_positive + extended.sum()) / 2.0
    negative_normalizer = len(score) - positive_normalizer
    if negative_normalizer <= 0:
        raise ValueError("Range-AUC negative normalizer must be positive")

    for index, threshold in enumerate(thresholds, start=1):
        prediction = score >= threshold
        true_positive = float(np.dot(extended, prediction))
        predicted = int(prediction.sum())
        false_positive = predicted - true_positive
        recall = min(true_positive / positive_normalizer, 1.0)
        detected = sum(
            bool(prediction[start:end + 1].any())
            for start, end in ranges
        )
        true_positive_rate = recall * detected / len(ranges)
        false_positive_rate = false_positive / negative_normalizer
        curve[index] = [true_positive_rate, false_positive_rate]
        precisions[index] = true_positive / predicted

    curve[-1] = [1.0, 1.0]
    roc_area = np.dot(
        np.diff(curve[:, 1]),
        (curve[1:, 0] + curve[:-1, 0]) / 2.0,
    )
    pr_area = np.dot(
        np.diff(curve[:-1, 0]),
        (precisions[1:] + precisions[:-1]) / 2.0,
    )
    return float(roc_area), float(pr_area)


def _official_vus_curve(
    labels_original: np.ndarray,
    score: np.ndarray,
    window_size: int,
) -> Tuple[float, float]:
    labels_original, score = _validate_range_inputs(labels_original, score)
    if window_size < 0:
        raise ValueError("windowSize cannot be negative")

    original_positive = float(labels_original.sum())
    original_ranges = range_convers_new(labels_original)
    max_sequences = _merged_extended_sequences(
        original_ranges,
        len(labels_original),
        window_size,
    )
    thresholds = _official_thresholds(score)
    predicted_counts = np.asarray(
        [(score >= threshold).sum() for threshold in thresholds],
        dtype=np.float64,
    )
    roc_areas = []
    pr_areas = []

    for window in range(window_size + 1):
        extended = extend_positive_range(labels_original, window)
        sequences = _merged_extended_sequences(
            original_ranges,
            len(labels_original),
            window,
        )
        curve = np.zeros((OFFICIAL_VUS_THRESHOLD_COUNT + 2, 2))
        precisions = np.ones(OFFICIAL_VUS_THRESHOLD_COUNT + 1)

        for threshold_index, threshold in enumerate(thresholds, start=1):
            prediction = score >= threshold
            adjusted = extended.copy()
            existence = 0
            for start, end in sequences:
                adjusted[start:end + 1] = (
                    extended[start:end + 1]
                    * prediction[start:end + 1]
                )
                if prediction[start:end + 1].any():
                    existence += 1
            for start, end in original_ranges:
                adjusted[start:end + 1] = 1.0

            true_positive = 0.0
            adjusted_positive = 0.0
            for start, end in max_sequences:
                segment = slice(start, end + 1)
                true_positive += float(
                    np.dot(adjusted[segment], prediction[segment])
                )
                adjusted_positive += float(adjusted[segment].sum())

            predicted = predicted_counts[threshold_index - 1]
            false_positive = predicted - true_positive
            positive_normalizer = (
                original_positive + adjusted_positive
            ) / 2.0
            negative_normalizer = len(score) - positive_normalizer
            if negative_normalizer <= 0:
                raise ValueError("VUS negative normalizer must be positive")
            recall = min(true_positive / positive_normalizer, 1.0)
            true_positive_rate = recall * existence / len(sequences)
            false_positive_rate = false_positive / negative_normalizer
            curve[threshold_index] = [
                true_positive_rate,
                false_positive_rate,
            ]
            precisions[threshold_index] = true_positive / predicted

        curve[-1] = [1.0, 1.0]
        roc_areas.append(
            float(
                np.dot(
                    np.diff(curve[:, 1]),
                    (curve[1:, 0] + curve[:-1, 0]) / 2.0,
                )
            )
        )
        pr_areas.append(
            float(np.dot(np.diff(curve[:-1, 0]), precisions[1:]))
        )

    return float(np.mean(roc_areas)), float(np.mean(pr_areas))


def RangeAUC_vectorized(labels_original: np.ndarray, score: np.ndarray,
                        window: int) -> Tuple[float, float]:
    """Compute the official reference Range-AUC pair."""
    return _range_curve(labels_original, score, window)


def RangeAUC_volume(labels_original: np.ndarray, score: np.ndarray,
                    windowSize: int) -> Tuple[float, float]:
    """Compute official VUS over every integer window in ``[0, windowSize]``."""
    return _official_vus_curve(labels_original, score, int(windowSize))


def evaluate_metrics(y_score: np.ndarray, y_true: np.ndarray,
                     slidingWindow: Optional[int] = None) -> Dict[str, float]:
    """Compute the four range-aware metrics used by the formal report."""
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).ravel()
    if y_score.size == 0:
        raise ValueError("Metric inputs cannot be empty")
    if y_score.shape != y_true.shape:
        raise ValueError("Scores and labels must have the same length")
    if not np.isfinite(y_score).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary values in {0, 1}")
    if slidingWindow is None:
        events = convert_vector_to_events(y_true)
        lengths = [end - start for start, end in events]
        slidingWindow = int(np.mean(lengths)) if lengths else 100
    if slidingWindow < 0:
        raise ValueError("slidingWindow cannot be negative")
    r_auc_roc, r_auc_pr = RangeAUC_vectorized(
        y_true, y_score, int(slidingWindow)
    )
    vus_roc, vus_pr = RangeAUC_volume(
        y_true, y_score, int(2 * slidingWindow)
    )
    return {
        "r_auc_roc": r_auc_roc,
        "r_auc_pr": r_auc_pr,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
    }
