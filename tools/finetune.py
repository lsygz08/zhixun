"""模型微调模块（当前为空壳 + 接口）。

当前 finetune 用"最小二乘重拟合"模拟微调，保证闭环能空转跑通；
真实微调接口见 real_finetune()（预留，接入 PyTorch 时实现）。

故障注入（问题三·1）：config 的 faults.inject_finetune_failure 为 True 时，
finetune() 抛出 FinetuneError，由调用方捕获后"降档重试"，仍失败则放弃候选。
"""
import base64
import json
import os
import pickle
import time

import numpy as np

from core import cfg_get, load_config
from core.monitor import impute


class FinetuneError(RuntimeError):
    """微调失败异常（用于故障注入与降级）。"""


def fit_linear(X, y, task_type="classification"):
    """用最小二乘拟合线性模型。返回 (权重, 偏置)。

    # modified：分类标签用 ±1（2y-1）使截距居中于 0、阈值 0 判定正确；
    # new：回归任务直接拟合原始 y（不做 ±1 映射）。
    """
    X = impute(X)
    y = np.asarray(y, dtype=float)
    y_signed = y if task_type == "regression" else 2.0 * y - 1.0
    Xb = np.column_stack([X, np.ones(X.shape[0])])
    coef, *_ = np.linalg.lstsq(Xb, y_signed, rcond=None)
    return coef[:-1], float(coef[-1])


_GBDT_CACHE = {}


def fit_gbdt(X, y, seed=None):
    """# new：GBDT 分类器（sklearn）。sklearn 不可用抛 ImportError，调用方回退线性。"""
    from sklearn.ensemble import GradientBoostingClassifier
    X = impute(X)
    y = np.asarray(y, dtype=int)
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                     random_state=int(seed or 42))
    clf.fit(X, y)
    return clf


def _serialize_gbdt(clf):
    """sklearn 对象 -> base64 字符串（可入 JSON 产物）。"""
    return base64.b64encode(pickle.dumps(clf)).decode("ascii")


def _load_gbdt(model):
    """从模型 dict 反序列化 GBDT（带缓存）。"""
    s = model.get("gbdt")
    if s not in _GBDT_CACHE:
        _GBDT_CACHE[s] = pickle.loads(base64.b64decode(s.encode("ascii")))
    return _GBDT_CACHE[s]


def predict(model, X):
    """模型推理：返回 (标签, 分数)。回归任务时标签=预测值（连续）。"""
    X = impute(X)
    if model.get("kind") == "gbdt":
        clf = _load_gbdt(model)
        score = clf.predict_proba(X)[:, 1]
        label = (score > 0.5).astype(int)
        return label, score
    w = np.asarray(model["weights"], dtype=float)
    b = float(model["bias"])
    score = X @ w + b
    if model.get("task_type") == "regression":
        return score, score
    label = (score > 0.0).astype(int)
    return label, score


def make_initial_model(segment, config=None):
    """# modified：基于基线数据段构造初始模型（linear / gbdt，model.algorithm 控制）。"""
    config = config or load_config()
    algorithm = cfg_get(config, "model.algorithm", "linear")
    task_type = cfg_get(config, "task.type", "classification")
    base = {
        "version": "v0",
        "kind": algorithm,
        "task_type": task_type,
        "metric": {},                      # 由 gate.compute_metrics 回填
        "params": {"C": 1.0, "gamma": 0.1},
        "artifact": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if algorithm == "gbdt" and task_type == "classification":
        try:
            clf = fit_gbdt(segment["X"], segment["y"], cfg_get(config, "run.seed", 42))
            base["gbdt"] = _serialize_gbdt(clf)
            return base
        except ImportError:
            print("[finetune] scikit-learn 不可用，回退线性模型")
            base["kind"] = "linear"
    if algorithm == "gbdt" and task_type != "classification":
        print("[finetune] GBDT 暂仅支持分类，回归回退线性模型")
        base["kind"] = "linear"
    w, b = fit_linear(segment["X"], segment["y"], task_type)
    base["weights"] = w.tolist()
    base["bias"] = b
    return base


def bump_version(version):
    """版本号 +1：v0 -> v1 -> v2 ..."""
    try:
        return "v" + str(int(str(version).lstrip("v")) + 1)
    except Exception:
        return "v1"


def finetune(base_model, segment, params=None, downgrade=False, seed=None, config=None):
    """模拟微调：打印"模拟微调完成"，返回新候选模型（空壳 + 接口）。

    参数:
        base_model: 基础模型 dict。
        segment: 当前数据段（用于重拟合，模拟学到新分布）。
        params: 超参数建议（当前只记录，不真正生效）。
        downgrade: True 表示降档重试（减少扰动幅度，模拟减小批量/学习率）。
        seed: 随机种子。
        config: 可选，配置字典（用于读取故障注入开关）。
    返回:
        候选模型 dict（含 weights / bias，metric 由 gate 回填）。

    故障注入：config 的 faults.inject_finetune_failure 为 True 时抛 FinetuneError。
    """
    config = config or load_config()

    # 故障注入（问题三·1）：模拟微调失败
    if cfg_get(config, "faults.inject_finetune_failure", False):
        raise FinetuneError("模拟：微调失败（演示降档重试）。")

    tag = "降档" if downgrade else ""
    print(f"[finetune] 模拟{tag}微调完成，参数={params}")
    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 32)

    kind = base_model.get("kind", "linear")
    if kind == "gbdt":
        try:
            clf = fit_gbdt(segment["X"], segment["y"], seed)
        except ImportError:
            raise FinetuneError("scikit-learn 不可用，GBDT 微调失败")
        return {
            "version": bump_version(base_model.get("version", "v0")),
            "kind": "gbdt",
            "gbdt": _serialize_gbdt(clf),
            "metric": {},
            "params": dict(params or {}),
            "artifact": None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # 线性：用当前段重拟合，模拟"学到了新分布"；加一点扰动模拟随机性
    w_fit, b_fit = fit_linear(segment["X"], segment["y"],
                              base_model.get("task_type", "classification"))
    rng = np.random.default_rng(seed)
    noise_scale = 0.01 if downgrade else 0.05
    w = w_fit + rng.normal(0.0, noise_scale, size=w_fit.shape)
    b = b_fit + rng.normal(0.0, noise_scale)

    return {
        "version": bump_version(base_model.get("version", "v0")),
        "kind": "linear",
        "task_type": base_model.get("task_type", "classification"),
        "weights": w.tolist(),
        "bias": float(b),
        "metric": {},                      # 由 gate 回填
        "params": dict(params or {}),
        "artifact": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def real_finetune(base_model, segment, params):
    """预留：真实微调接口（接入 PyTorch 时实现）。当前 raise NotImplementedError。"""
    raise NotImplementedError("真实微调接口尚未接入。")


def save_artifact(model, artifacts_dir):
    """把模型存成 JSON 产物，返回产物路径。"""
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"model_{model['version']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    return path
