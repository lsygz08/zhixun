"""实验优化模块。

根据归因卡 drift_type 分发动作（动作与判断严格对齐）：
- data_drift    -> resample_or_calibrate（先重采样，再对当前模型做 Platt 概率校准）
- concept_drift -> finetune（模拟微调）
- no_drift      -> no_action

包含：
- suggest()：归因卡 -> 动作分发
- execute()：动作 -> 候选模型（真正执行）
- resample_and_calibrate()：重采样 + Platt scaling 校准
- rule_fallback_plan()：归因模块异常时的纯规则兜底
- cost_benefit_check()：成本收益判断
"""
import time

import numpy as np

from core import cfg_get, load_config
from tools import finetune

_GRID_CALLS = [0]


def action_cost(action, config=None):
    """返回某动作的预估资源成本（供预算账本使用）。"""
    config = config or load_config()
    if action == "finetune":
        return {
            "gpu_hours": cfg_get(config, "cost.finetune_gpu", 0.2),
            "tokens": cfg_get(config, "cost.finetune_tokens", 1000),
        }
    if action == "resample_or_calibrate":
        return {"labels": cfg_get(config, "cost.resample_labels", 200)}
    if action == "llm":   # new：大模型调用按 [cost].llm_tokens 扣费
        return {"tokens": cfg_get(config, "cost.llm_tokens", 500)}
    return {}


def grid_suggest(config=None, index=None):
    """网格搜索建议器：从 config.yaml 的 grid 中轮询选取一组参数。"""
    config = config or load_config()
    grid = cfg_get(config, "optimize.grid", {}) or {}
    if not grid:
        return {}
    idx = index if index is not None else _GRID_CALLS[0]
    _GRID_CALLS[0] += 1
    params = {}
    for key, values in grid.items():
        values = values if isinstance(values, list) else [values]
        if values:
            params[key] = values[idx % len(values)]
    return params


def bayesian_suggest(config=None):
    """预留：贝叶斯优化接口。当前返回 None。"""
    return None


def suggest(card, config=None, memory=None):
    """根据归因卡 drift_type 分发动作（动作与判断严格对齐）。

    参数:
        card: diagnose.attribute 输出的归因卡。
        config: 可选，配置字典。
    返回:
        plan 字典：action / params / reason / expected_gain / card。
    """
    config = config or load_config()
    drift_type = card.get("drift_type", "no_drift")
    # 显式按 drift_type 分发动作
    action_map = {
        "data_drift": "resample_or_calibrate",
        "concept_drift": "finetune",
        "no_drift": "no_action",
        "none": "no_action",   # 兼容旧值
    }
    rule_action = action_map.get(drift_type, "no_action")

    # 策略层自改进：有历史记忆时，优先采用历史成功率更高的动作
    action = rule_action
    strategy_note = ""
    if memory and rule_action != "no_action":
        from core import strategy
        mem_action, mem_rate = strategy.suggest_action(memory, drift_type, rule_action)
        if mem_rate >= 0.0:
            action = mem_action
            strategy_note = f"（策略记忆采用 {action}，历史成功率 {mem_rate:.0%}）"

    if action == "no_action":
        return {
            "action": "no_action",
            "params": {},
            "reason": "无漂移，无需动作。",
            "expected_gain": 0.0,
            "card": card,
        }
    params = grid_suggest(config)
    reason = (f"归因卡 drift_type={drift_type}（严重程度 {card.get('severity')}），"
              f"分发动作：{action}。{strategy_note}")
    expected_gain = card.get("expected_gain",
                             {"low": 0.01, "medium": 0.03, "high": 0.06}.get(card.get("severity"), 0.01))
    return {
        "action": action,
        "params": params,
        "reason": reason,
        "expected_gain": expected_gain,
        "card": card,
    }


def cost_benefit_check(plan, budget, config=None):
    """调大模型 / 重训练前，先判断预算是否值得。"""
    config = config or load_config()
    action = plan.get("action", "no_action")
    if action == "no_action":
        return True, "无需动作，不产生额外成本。"
    cost = action_cost(action, config)
    if not budget.check(cost):
        return False, (f"预算不足：动作 {action} 需要 {cost}，"
                       f"当前已用 {budget.summary()['used']}。")
    gain = plan.get("expected_gain", 0.0)
    min_gain = cfg_get(config, "optimize.min_expected_gain", 0.01)
    if gain < min_gain:
        return False, f"预期增益 {gain} 低于门槛 {min_gain}，不值得执行。"
    return True, f"预期增益 {gain}，成本 {cost}，值得执行。"


def rule_fallback_plan(stats, config=None):
    """纯规则策略兜底（问题三·2）：归因模块异常时，直接按硬阈值输出 plan。"""
    config = config or load_config()
    psi = float(stats.get("psi", 0.0))
    ks_p = float(stats.get("ks_pvalue", 1.0))
    miss = float(stats.get("missing_rate", 0.0))
    pred = float(stats.get("pred_shift", 0.0))

    psi_t = cfg_get(config, "drift.psi_trigger", 0.20)
    ks_a = cfg_get(config, "drift.ks_alpha", 0.05)
    miss_t = cfg_get(config, "drift.missing_trigger", 0.15)
    pred_t = cfg_get(config, "drift.pred_shift_trigger", 0.20)

    feature_drift = (psi >= psi_t) and (ks_p < ks_a)
    concept_drift = pred >= pred_t

    if feature_drift or miss >= miss_t:
        action = "resample_or_calibrate"
        reason = "纯规则兜底：特征分布/缺失率超阈值，按数据漂移处理（重采样/校准）。"
        expected_gain = 0.02
    elif concept_drift:
        action = "finetune"
        reason = "纯规则兜底：预测分布偏移超阈值，按概念漂移处理（微调）。"
        expected_gain = 0.04
    else:
        action = "no_action"
        reason = "纯规则兜底：未达触发阈值，不动作。"
        expected_gain = 0.0

    params = grid_suggest(config) if action != "no_action" else {}
    return {
        "action": action,
        "params": params,
        "reason": reason,
        "expected_gain": expected_gain,
        "card": None,
        "fallback": True,
    }


def _resample_balanced(X, y, seed):
    """类别均衡重采样：每类都过采样到多数类数量（缓解标签不平衡）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y            # 只有一类，无法均衡，原样返回
    n_major = int(counts.max())
    Xs, ys = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        idx = rng.choice(idx, size=n_major, replace=True)
        Xs.append(X[idx])
        ys.append(y[idx])
    return np.vstack(Xs), np.concatenate(ys)


def _fit_platt_numpy(scores, y):
    """numpy 兜底：牛顿法拟合二参逻辑回归（Platt scaling）。"""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    A, B = 1.0, 0.0
    for _ in range(30):
        z = np.clip(A * scores + B, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        err = p - y
        g = np.array([np.mean(err * scores), np.mean(err)])
        H = np.array([[np.mean(w * scores * scores), np.mean(w * scores)],
                      [np.mean(w * scores), np.mean(w)]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        A -= float(step[0])
        B -= float(step[1])
    return float(A), float(B)


def _fit_platt(scores, y):
    """Platt scaling：拟合 P(y=1|s)=sigmoid(A*s+B)。优先 sklearn，兜底 numpy。"""
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.0, 0.0
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(scores, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return _fit_platt_numpy(scores.ravel(), y)


def resample_and_calibrate(base_model, segment, seed=None, config=None):
    """data_drift 动作：先重采样（类别均衡）再对当前模型做 Platt 概率校准。

    1. 重采样：把漂移段按类别均衡过采样，缓解标签不平衡；
    2. 校准：用当前模型在重采样集上的分数拟合 Platt 参数 (A, B)，
       P(y=1|s)=sigmoid(A*s+B)；把 0.5 概率阈值折进 bias，使现有 predict
       的 score>0 判定等价于校准后的概率阈值。
    权重保持不变（数据漂移用"校准"而非"重训"来修正概率漂移）。
    """
    config = config or load_config()
    if seed is None:
        seed = int(cfg_get(config, "run.seed", 42))

    X_res, y_res = _resample_balanced(segment["X"], segment["y"], seed)
    _, scores = finetune.predict(base_model, X_res)   # 当前模型打分
    A, B = _fit_platt(scores, y_res)

    threshold_shift = B / A if A > 1e-8 else 0.0
    new_bias = float(base_model.get("bias", 0.0)) + threshold_shift
    print(f"[optimize] 重采样 + Platt 校准完成，calib A={A:.4f} B={B:.4f}，阈值平移 {threshold_shift:.4f}")

    return {
        "version": finetune.bump_version(base_model.get("version", "v0")),
        "kind": base_model.get("kind", "linear"),
        "weights": list(base_model.get("weights", [])),
        "bias": float(new_bias),
        "calibrated": True,
        "calib_a": float(A),
        "calib_b": float(B),
        "metric": {},                        # 由 gate 回填
        "params": {"action": "resample_or_calibrate"},
        "artifact": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def execute(plan, base_model, segment, seed=None, downgrade=False, config=None):
    """按 plan['action'] 分发执行，生成候选模型。

    - finetune              -> finetune.finetune（模拟微调，支持降档）
    - resample_or_calibrate -> resample_and_calibrate（重采样 + Platt 校准）
    - no_action             -> 返回 None
    """
    action = plan.get("action", "no_action")
    if action == "finetune":
        return finetune.finetune(base_model, segment, plan.get("params"),
                                 downgrade=downgrade, seed=seed, config=config)
    if action == "resample_or_calibrate":
        return resample_and_calibrate(base_model, segment, seed=seed, config=config)
    return None
