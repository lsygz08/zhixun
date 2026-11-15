"""诊断归因模块（对齐官方 3.3：归因卡固定 7 字段）。

归因卡固定 7 字段：
漂移类型 / 受影响子群 / 证据指标 / 置信度 / 建议动作 / 预估收益 / 预估成本。
（另保留 severity / reason / level 三个扩展字段）

判定规则：
- PSI 高 且 KS 检验显著（p < ks_alpha）-> data_drift -> resample_or_calibrate
- PSI 低 但 pred_shift 高 -> concept_drift -> finetune
- 两者都低 -> no_drift -> no_action
- 缺失率超阈值 也归入 data_drift
"""
from core import cfg_get, load_config


class DiagnoseError(RuntimeError):
    """归因模块异常（用于故障注入与降级）。"""


def _action_cost(action, config):
    """# new：按建议动作给出预估成本（读 cost.*，不 import optimize 避免耦合）。"""
    if action == "finetune":
        return {"gpu_hours": cfg_get(config, "cost.finetune_gpu", 0.2),
                "tokens": cfg_get(config, "cost.finetune_tokens", 1000)}
    if action == "resample_or_calibrate":
        return {"labels": cfg_get(config, "cost.resample_labels", 200)}
    return {}


def _card(dtype, severity, confidence, action, reason, affected, evidence, gain, cost, level):
    """# modified：组装七字段归因卡（+3 个扩展字段）。"""
    return {
        "drift_type": dtype,                    # 漂移类型
        "affected_subgroups": affected,         # 受影响子群
        "evidence": evidence,                   # 证据指标
        "confidence": round(float(confidence), 3),  # 置信度
        "suggested_action": action,             # 建议动作
        "expected_gain": gain,                  # 预估收益
        "expected_cost": cost,                  # 预估成本
        "severity": severity,                   # 扩展：严重程度
        "reason": reason,                       # 扩展：理由文本
        "level": level,                         # 扩展：监测等级
    }


def _rule_attribute(stats, config):
    """纯规则归因逻辑（不含故障注入）。"""
    psi = float(stats.get("psi", 0.0))
    ks = float(stats.get("ks", 0.0))
    ks_p = float(stats.get("ks_pvalue", 1.0))
    miss = float(stats.get("missing_rate", 0.0))
    pred = float(stats.get("pred_shift", 0.0))

    psi_t = cfg_get(config, "drift.psi_trigger", 0.20)
    ks_a = cfg_get(config, "drift.ks_alpha", 0.05)
    miss_t = cfg_get(config, "drift.missing_trigger", 0.15)
    pred_t = cfg_get(config, "drift.pred_shift_trigger", 0.20)
    psi_w = cfg_get(config, "drift.psi_warn", 0.10)
    miss_w = cfg_get(config, "drift.missing_warning", 0.05)
    pred_w = cfg_get(config, "drift.pred_shift_warning", 0.10)

    # 漂移类型判定（官方 3.3：概念漂移优先；特征漂移用 PSI「或」KS）
    concept_drift = pred >= pred_t                      # 预测分布偏移高（映射关系变）
    feature_drift = (psi >= psi_t) or (ks_p < ks_a)     # PSI 高 或 KS 检验显著

    if concept_drift:
        dtype = "concept_drift"
        action = "finetune"
        reason = (f"预测分布偏移 pred_shift={pred:.3f}(阈值{pred_t})，"
                  f"输入-标签映射关系改变，判定为概念漂移，建议微调。")
    elif feature_drift or (miss >= miss_t):
        dtype = "data_drift"
        action = "resample_or_calibrate"
        reason = (f"特征分布发生漂移：PSI={psi:.3f}(阈值{psi_t})、KS p={ks_p:.4g}(阈值{ks_a})"
                  + (f"、缺失率={miss:.3f}(阈值{miss_t})" if miss >= miss_t else "")
                  + "，判定为数据漂移，建议重采样/校准。")
    else:
        dtype = "no_drift"
        action = "no_action"
        reason = "各项指标均未达触发阈值，未检测到显著漂移。"

    # 受影响子群：漂移最大的若干特征（data_drift），概念漂移则为整体映射
    if dtype == "data_drift":
        pp = stats.get("per_feature_psi", {})
        affected = sorted(pp, key=lambda k: pp.get(k, 0.0), reverse=True)[:3] if pp else []
    elif dtype == "concept_drift":
        affected = ["整体输入-输出映射关系"]
    else:
        affected = []

    # 证据指标
    evidence = {
        "psi": round(psi, 6), "ks": round(ks, 6), "ks_pvalue": round(ks_p, 6),
        "missing_rate": round(miss, 6), "pred_shift": round(pred, 6),
        "thresholds": {"psi_trigger": psi_t, "ks_alpha": ks_a,
                       "missing_trigger": miss_t, "pred_shift_trigger": pred_t},
    }

    # 严重程度分级
    def _ratio(v, t):
        return (v / t) if t and t > 0 else 0.0
    ks_ratio = (ks_a / ks_p) if ks_p > 0 else 0.0
    ratios = [_ratio(psi, psi_t), ks_ratio, _ratio(miss, miss_t), _ratio(pred, pred_t)]
    max_ratio = max(ratios)
    n_signals = sum([psi >= psi_w, miss >= miss_w, pred >= pred_w])

    triggered = bool(stats.get("triggered", False))
    if not triggered:
        severity = "low"
    elif max_ratio >= 2.0:
        severity = "high"
    else:
        severity = "medium"

    # 置信度：超阈值越多越接近 1（不写死）
    if triggered:
        confidence = min(0.99, 1.0 - 0.5 / max(1e-6, max_ratio))
    else:
        confidence = min(0.9, 0.5 + 0.08 * n_signals)

    # 预估收益 / 预估成本
    gain = {"low": 0.01, "medium": 0.03, "high": 0.06}.get(severity, 0.0)
    cost = _action_cost(action, config)

    return _card(dtype, severity, confidence, action, reason, affected, evidence,
                 round(gain, 3), cost, stats.get("level", "normal"))


def attribute(stats, config=None):
    """规则版归因。故障注入：faults.inject_diagnose_failure 为 True 时抛 DiagnoseError。"""
    config = config or load_config()
    if cfg_get(config, "faults.inject_diagnose_failure", False):
        raise DiagnoseError("模拟：归因模块异常（演示降级）。")
    return _rule_attribute(stats, config)


def rule_fallback(stats, config=None):
    """纯规则归因（不含故障注入），供需要硬兜底时调用。"""
    return _rule_attribute(stats, config or load_config())


def llm_attribute(stats, config=None):
    """预留：调用大模型做归因。当前默认关闭，返回 None。"""
    return None
