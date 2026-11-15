"""评估门禁模块（对齐官方 3.6 三层门禁 + 多指标一致性 warning）。

三层门禁（全部通过才允许上线）：
1. 绝对指标：候选在验证集上的主指标 >= 当前线上模型 + min_gain；
2. 抗漂移能力：候选在漂移回放集上的主指标 >= 静态基线；
3. 成本合规：三类消耗在预算余额内。

多指标一致性（f1/auc 相对回退）降级为 warning，不再作为拒判条件。
"""
import numpy as np

from core import cfg_get, load_config
from tools import finetune


class GateError(RuntimeError):
    """门禁模块异常（用于故障注入与降级）。"""


def platform_metrics(model_mlp, data):
    """# new：官方评测口径指标（与官方 baselines.train.metrics 完全一致）。

    分类：accuracy / macro_f1（softmax argmax）；回归：mae / rmse（越低越好）。
    用于平台数据上的门禁/记录，替代演示用的二分类 compute_metrics。
    """
    predicted = model_mlp.predict(data["x"])
    expected = data["y"]
    if model_mlp.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == c) & (expected == c)) /
              (np.sum(labels == c) + np.sum(expected == c)) for c in observed]
        return {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
    error = predicted - expected
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def _rankdata(a):
    """简易 rank（升序名次，从 1 开始；忽略并列平均，骨架够用）。"""
    a = np.asarray(a)
    ranks = np.empty(len(a), dtype=float)
    sorter = np.argsort(a)
    ranks[sorter] = np.arange(1, len(a) + 1)
    return ranks


def _auc(y_true, scores):
    """计算二分类 AUC（优先 sklearn，缺失时用秩近似）。"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        ranks = _rankdata(scores)
        sum_pos = ranks[y_true == 1].sum()
        n_pos, n_neg = len(pos), len(neg)
        return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true, y_pred, scores=None, task_type="classification"):
    """计算指标：分类 -> accuracy/f1/auc；回归 -> rmse/mae/r2。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if task_type == "regression":
        err = y_pred - y_true
        rmse = float(np.sqrt(np.mean(err ** 2))) if len(y_true) else 0.0
        mae = float(np.mean(np.abs(err))) if len(y_true) else 0.0
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    if len(y_true) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.5}
    accuracy = float(np.mean(y_true == y_pred))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = _auc(y_true, scores if scores is not None else y_pred)
    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4), "auc": round(auc, 4)}


def evaluate_model(model, segment):
    """在 segment 上评估 model（只读、无副作用），返回指标 dict。"""
    pred, scores = finetune.predict(model, segment["X"])
    return compute_metrics(segment["y"], pred, scores,
                           model.get("task_type", "classification"))


_LOWER_BETTER = ("rmse", "mae")


def _gain(cand, base, metric):
    """# new：指标增益——越高越好 -> cand-base；越低越好(rmse/mae) -> base-cand。"""
    if metric in _LOWER_BETTER:
        return float(base) - float(cand)
    return float(cand) - float(base)


def _relative_drop(base, cand):
    """相对回退比例 = (base - cand) / base。base<=0 时视作无回退，返回 0.0。"""
    base = float(base)
    cand = float(cand)
    if base <= 1e-12:
        return 0.0
    return (base - cand) / base


def _pct(x):
    """把比例格式化成百分比字符串。"""
    return f"{x * 100:.1f}%"


def evaluate(candidate, baseline, budget, segment=None, config=None):
    """三层门禁检查（官方 3.6），返回是否通过 + 理由。

    参数:
        candidate: 候选模型 dict（含 metric）。
        baseline: 当前线上模型 dict（含 metric 与 baseline_metrics 静态基线参照）。
        budget: BudgetTracker 实例。
        segment: 漂移回放集（此处用当前段代替；缺省时第二层跳过）。
        config: 可选，配置字典。
    返回:
        {"pass", "checks", "warnings", "reason", ...}。

    故障注入：faults.inject_gate_failure 为 True 时抛 GateError（由调用方降级）。
    """
    config = config or load_config()

    # 故障注入（问题三·3）：门禁模块异常
    if cfg_get(config, "faults.inject_gate_failure", False):
        raise GateError("模拟：门禁模块异常（演示降级）。")

    cand_m = candidate.get("metric") or {}
    base_m = baseline.get("metric") or {}
    primary_metric = cfg_get(config, "task.primary_metric", "accuracy")
    cand_acc = float(cand_m.get(primary_metric, 0.0))
    base_acc = float(base_m.get(primary_metric, 0.0))

    min_gain = cfg_get(config, "gate.min_gain", 0.0)
    reg_margin = cfg_get(config, "gate.severe_regression_margin", 0.15)

    # 第一层：绝对指标（按主指标方向比较：越大越好 / rmse、mae 越小越好）
    check1 = _gain(cand_acc, base_acc, primary_metric) >= min_gain

    # 第二层：抗漂移能力（候选在漂移回放集上 vs 静态基线）
    static_m = baseline.get("baseline_metrics") or base_m or {}
    static_acc = float(static_m.get(primary_metric, 0.0))
    replay_perf = None
    if segment is not None:
        replay_perf = evaluate_model(candidate, segment)
        replay_acc = float(replay_perf.get(primary_metric, 0.0))
        check2 = _gain(replay_acc, static_acc, primary_metric) >= 0.0
    else:
        replay_acc = None
        check2 = True   # 无回放集时跳过第二层（降级）

    # 第三层：成本合规
    check3 = budget.check({})

    checks = {
        "metric_not_worse_than_baseline": bool(check1),
        "anti_drift_not_worse_than_static_baseline": bool(check2),
        "within_budget": bool(check3),
    }
    passed = all(checks.values())

    # 多指标一致性 -> warning（降级，不再拒判）
    key_metrics = cfg_get(config, "task.key_metrics", ["f1", "auc"]) or ["f1", "auc"]
    drops = {m: _relative_drop(float(base_m.get(m, 0.0)), float(cand_m.get(m, 0.0)))
             for m in key_metrics}
    warnings = []
    for m, d in drops.items():
        if d > reg_margin:
            warnings.append(f"关键指标 {m} 回退 {_pct(d)}，超过阈值 {_pct(reg_margin)}")
    if _gain(cand_acc, base_acc, primary_metric) > 0:
        for m in key_metrics:
            if drops[m] > 0:
                warnings.append(f"指标方向不一致：{primary_metric} 改善，但 {m} 回退 {_pct(drops[m])}")

    # 组装理由
    if passed:
        reason = (f"候选指标 {round(cand_acc, 4)} 达标（线上 {round(base_acc, 4)}），"
                  f"回放集抗漂移达标，预算充足，允许上线。")
    else:
        fail_parts = []
        if not check1:
            fail_parts.append(f"绝对指标不达标（候选 {round(cand_acc, 4)} vs 线上 {round(base_acc, 4)}+增益 {min_gain}）")
        if not check2:
            fail_parts.append(f"抗漂移能力不达标（回放集 {round(replay_acc, 4)} vs 静态基线 {round(static_acc, 4)}）")
        if not check3:
            fail_parts.append("预算不足")
        reason = "门禁未通过：" + "；".join(fail_parts) + "。"
    if warnings:
        reason += " " + "；".join(warnings) + "。"

    return {
        "pass": passed,
        "checks": checks,
        "warnings": warnings,
        "key_metric_drops": {m: round(d, 4) for m, d in drops.items()},
        "reason": reason,
        "candidate_metric": cand_m,
        "baseline_metric": base_m,
        "replay_metric": replay_perf,
    }


def fallback_evaluate(candidate, segment, config=None):
    """降级路径（问题三·4）：评测接口不可用时，用本地漂移回放集评估，标记"待补测"。"""
    pred, scores = finetune.predict(candidate, segment["X"])
    m = compute_metrics(segment["y"], pred, scores, candidate.get("task_type", "classification"))
    return {
        "pass": True,
        "checks": {
            "metric_not_worse_than_baseline": True,
            "anti_drift_not_worse_than_static_baseline": True,
            "within_budget": True,
        },
        "warnings": [],
        "reason": "评测接口不可用，已用本地漂移回放集评估，本次评测标记为【待补测】。",
        "pending_retest": True,
        "candidate_metric": m,
        "baseline_metric": {},
    }


def should_act(diagnosis, current_model, segment, config=None):
    """该不该动闸门：判断这次漂移是否值得处理。只判断，绝无副作用。

    三层短路（按顺序）：
    1. 漂移轻微/未达触发阈值（level != trigger）-> 不动；
    2. 漂移严重但当前模型在 segment 上性能未明显退化 -> 不动；
    3. 漂移严重且性能确实退化 -> 动。
    """
    config = config or load_config()
    level = diagnosis.get("level", "normal")

    # 第一层：漂移轻微（未达触发阈值）-> 不动
    if level == "warning":
        return False, "漂移仅达预警线，未达触发阈值，保持基线"
    if level == "normal":
        return False, "未检测到触发级漂移，保持基线"

    # 第二层：评估当前线上模型（不是任何候选），性能未退化 -> 不动
    current_perf = evaluate_model(current_model, segment)
    baseline = current_model.get("baseline_metrics") or current_model.get("metric") or {}
    primary_metric = cfg_get(config, "task.primary_metric", "accuracy")
    base_acc = float(baseline.get(primary_metric, 0.0))
    cur_acc = float(current_perf.get(primary_metric, 0.0))
    floor_ratio = cfg_get(config, "gate.accuracy_floor_ratio", 0.95)
    # 越高越好的指标：退化 = cur < base*ratio；rmse/mae 越低越好：退化 = cur > base/ratio
    if primary_metric in _LOWER_BETTER:
        degraded = base_acc > 1e-12 and cur_acc > base_acc / max(floor_ratio, 1e-6)
    else:
        degraded = base_acc > 1e-12 and cur_acc < base_acc * floor_ratio
    if not degraded:
        return False, (f"当前模型在此漂移数据上性能未退化"
                       f"（当前{cur_acc:.3f} vs 基线{base_acc:.3f}），无需动作")

    # 第三层：漂移触发且当前模型性能确实退化 -> 动
    return True, f"漂移触发且当前模型性能退化，建议{diagnosis.get('suggested_action', '')}"
