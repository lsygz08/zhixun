"""漂移监测模块（对齐官方 3.2 表2 六类监测对象）。

覆盖六类对象：
1. 数值特征分布：PSI、KS 检验（统计量 + p 值）
2. 类别特征分布：频率分布总变差距离
3. 数据质量：缺失率、越界率、重复率
4. 预测分布：预测分布偏移、置信度偏移、类别占比偏移
5. 延迟标签性能：滚动窗口主指标（本段模型准确率相对基线的下降）
6. 服务健康：推理延迟 / 失败率（读 segment.meta.service）

阈值分两档：预警只记录，触发才进归因；触发需连续 consecutive 个批次确认（默认 2）。
KS 触发采用官方口径：KS 检验 p < ks_alpha。
"""
import numpy as np
from collections import Counter

from core import cfg_get, load_config

_BASELINE = None
_CONSECUTIVE_TRIGGER = 0   # 连续触发批次计数


def impute(X):
    """用各列均值填充 NaN（返回副本，不改动原数组）。"""
    X = np.array(X, dtype=float, copy=True)
    if X.size == 0:
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_mean, inds[1])
    return X


def psi(expected, actual, bins=10, min_bin=1e-4):
    """计算两个一维分布的 PSI（Population Stability Index）。"""
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    lo = min(float(np.min(expected)), float(np.min(actual)))
    hi = max(float(np.max(expected)), float(np.max(actual)))
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e = np.histogram(expected, bins=edges)[0].astype(float) / len(expected)
    a = np.histogram(actual, bins=edges)[0].astype(float) / len(actual)
    e = np.clip(e, min_bin, None)
    a = np.clip(a, min_bin, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _ks_statistic_numpy(a, b):
    """numpy 兜底：两样本 KS 统计量（经验分布最大差）。"""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    combined = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, combined, side="right") / len(a)
    cdf_b = np.searchsorted(b, combined, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ks_pvalue_numpy(stat, n, m):
    """numpy 兜底：Kolmogorov 分布渐近近似，把 KS 统计量转成 p 值。"""
    if stat <= 0:
        return 1.0
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * stat
    s = 0.0
    for j in range(1, 60):
        term = (-1.0) ** (j - 1) * np.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, 2.0 * s)))


def ks_test(a, b):
    """KS 两样本检验，返回 (统计量, p 值)。优先 scipy，兜底 numpy 渐近近似。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        stat = _ks_statistic_numpy(a, b)
        return stat, _ks_pvalue_numpy(stat, len(a), len(b))


def missing_rate(X):
    """计算整体缺失率。"""
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return 0.0
    return float(np.isnan(X).mean())


def prediction_shift(ref_scores, cur_scores):
    """计算预测分布偏移（用预测分数的 PSI 近似）。"""
    if ref_scores is None or cur_scores is None:
        return 0.0
    try:
        return psi(ref_scores, cur_scores)
    except Exception:
        return float(abs(np.nanmean(cur_scores) - np.nanmean(ref_scores)))


def _total_variation(p, q):
    """# new：两个频率分布的总变差距离。"""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def categorical_drift(ref_X, X, max_unique=20):
    """# new：类别特征频率分布的总变差距离（低基数特征视为类别特征）。"""
    drift = {}
    n = ref_X.shape[1]
    for j in range(n):
        cr = ref_X[:, j]
        cc = X[:, j]
        if len(np.unique(cr)) <= max_unique or len(np.unique(cc)) <= max_unique:
            pr = Counter(cr.tolist())
            pc = Counter(cc.tolist())
            pr = {k: v / len(cr) for k, v in pr.items()}
            pc = {k: v / len(cc) for k, v in pc.items()}
            drift[f"f{j}"] = round(_total_variation(pr, pc), 6)
    return drift


def out_of_range_rate(X, ref_min, ref_max):
    """# new：越界率——超出基线取值范围的元素比例。"""
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return 0.0
    return float(np.sum((X < ref_min) | (X > ref_max)) / X.size)


def duplicate_rate(X):
    """# new：重复率——重复行占比。"""
    X = np.asarray(X)
    if X.shape[0] == 0:
        return 0.0
    return float(1.0 - np.unique(X, axis=0).shape[0] / X.shape[0])


def _accuracy(y, pred):
    if y is None or pred is None or len(y) == 0:
        return None
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def set_baseline(segment):
    """设置参考（基线）数据段。"""
    global _BASELINE
    _BASELINE = segment


def reset_baseline():
    """清空基线（每次新的 run 调用一次），并重置连续触发计数。"""
    global _BASELINE, _CONSECUTIVE_TRIGGER
    _BASELINE = None
    _CONSECUTIVE_TRIGGER = 0


def _baseline_result(segment):
    """首次无参考时的返回：零漂移，标记为基线。"""
    return {
        "psi": 0.0, "ks": 0.0, "ks_pvalue": 1.0,
        "missing_rate": missing_rate(segment.get("X")),
        "pred_shift": 0.0,
        "categorical_drift": {},
        "out_of_range_rate": 0.0,
        "duplicate_rate": duplicate_rate(segment.get("X")),
        "confidence_shift": None,
        "class_proportion_shift": None,
        "label_perf_drop": None,
        "service_health": (segment.get("meta") or {}).get("service", {}) or {},
        "per_feature_psi": {}, "per_feature_ks": {}, "per_feature_ks_pvalue": {},
        "is_baseline": True, "level": "normal", "warned": False, "triggered": False,
        "consecutive_count": 0,
    }


def drift_stats(segment, reference=None, config=None):
    """计算当前数据段相对参考段的漂移统计，并做两档阈值 + 连续批次判定。

    返回统计字典，含六类监测对象的指标，以及
    level（normal|warning|trigger）/ warned / triggered / consecutive_count。
    """
    global _BASELINE, _CONSECUTIVE_TRIGGER
    config = config or load_config()

    if reference is None:
        reference = _BASELINE
    if reference is None:
        _BASELINE = segment
        return _baseline_result(segment)

    X = impute(segment.get("X"))
    ref_X = impute(reference.get("X"))

    per_feature_psi = {}
    per_feature_ks = {}
    per_feature_ks_p = {}
    n_feat = X.shape[1]
    for j in range(n_feat):
        per_feature_psi[f"f{j}"] = psi(ref_X[:, j], X[:, j])
        s, p = ks_test(ref_X[:, j], X[:, j])
        per_feature_ks[f"f{j}"] = s
        per_feature_ks_p[f"f{j}"] = p

    psi_val = float(np.nanmean(list(per_feature_psi.values())))
    ks_val = float(np.nanmax(list(per_feature_ks.values())))
    ks_p = float(np.nanmin(list(per_feature_ks_p.values())))
    miss_val = missing_rate(segment.get("X"))
    pred_val = prediction_shift(reference.get("scores"), segment.get("scores"))

    # new：其余五类对象
    cat_drift = categorical_drift(ref_X, X)
    oor = out_of_range_rate(X, np.min(ref_X, axis=0), np.max(ref_X, axis=0))
    dup = duplicate_rate(segment.get("X"))

    ref_scores = reference.get("scores")
    cur_scores = segment.get("scores")
    conf_shift = None
    if ref_scores is not None and cur_scores is not None:
        rs = np.asarray(ref_scores, dtype=float)
        cs = np.asarray(cur_scores, dtype=float)
        conf_shift = float(np.nanmean(np.abs(cs)) - np.nanmean(np.abs(rs))) if (rs.size and cs.size) else None

    prop_shift = None
    if reference.get("pred") is not None and segment.get("pred") is not None:
        rp = np.asarray(reference["pred"])
        cp = np.asarray(segment["pred"])
        if rp.size and cp.size:
            prop_shift = float(abs(np.mean(rp == 1) - np.mean(cp == 1)))

    cur_acc = _accuracy(segment.get("y"), segment.get("pred"))
    ref_acc = _accuracy(reference.get("y"), reference.get("pred"))
    label_perf_drop = (ref_acc - cur_acc) if (cur_acc is not None and ref_acc is not None) else None

    service = (segment.get("meta") or {}).get("service", {}) or {}

    psi_w = cfg_get(config, "drift.psi_warn", 0.10)
    psi_t = cfg_get(config, "drift.psi_trigger", 0.20)
    ks_a = cfg_get(config, "drift.ks_alpha", 0.05)
    miss_w = cfg_get(config, "drift.missing_warning", 0.05)
    miss_t = cfg_get(config, "drift.missing_trigger", 0.15)
    pred_w = cfg_get(config, "drift.pred_shift_warning", 0.10)
    pred_t = cfg_get(config, "drift.pred_shift_trigger", 0.20)

    raw_triggered = any([psi_val >= psi_t, ks_p < ks_a, miss_val >= miss_t, pred_val >= pred_t])
    raw_warned = any([psi_val >= psi_w, miss_val >= miss_w, pred_val >= pred_w])

    # new：连续批次确认
    if raw_triggered:
        _CONSECUTIVE_TRIGGER += 1
    else:
        _CONSECUTIVE_TRIGGER = 0
    consecutive = int(cfg_get(config, "drift.consecutive", 2))
    confirmed = raw_triggered and (_CONSECUTIVE_TRIGGER >= consecutive)

    if confirmed:
        level, triggered, warned = "trigger", True, True
    elif raw_triggered or raw_warned:
        level, triggered, warned = "warning", False, True
    else:
        level, triggered, warned = "normal", False, False

    return {
        "psi": round(psi_val, 6),
        "ks": round(ks_val, 6),
        "ks_pvalue": round(ks_p, 6),
        "missing_rate": round(miss_val, 6),
        "pred_shift": round(pred_val, 6),
        "categorical_drift": {k: round(v, 6) for k, v in cat_drift.items()},
        "out_of_range_rate": round(oor, 6),
        "duplicate_rate": round(dup, 6),
        "confidence_shift": round(conf_shift, 6) if conf_shift is not None else None,
        "class_proportion_shift": round(prop_shift, 6) if prop_shift is not None else None,
        "label_perf_drop": round(label_perf_drop, 6) if label_perf_drop is not None else None,
        "service_health": service,
        "per_feature_psi": {k: round(v, 6) for k, v in per_feature_psi.items()},
        "per_feature_ks": {k: round(v, 6) for k, v in per_feature_ks.items()},
        "per_feature_ks_pvalue": {k: round(v, 6) for k, v in per_feature_ks_p.items()},
        "is_baseline": False,
        "level": level,
        "warned": warned,
        "triggered": triggered,
        "consecutive_count": _CONSECUTIVE_TRIGGER,
    }
