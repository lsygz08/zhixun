"""数据加载模块：先用假数据跑通闭环，预留真实数据接口。

# modified：重写造数据逻辑，保证特征分布与标签分布自洽，指标不再互相矛盾。
新增两个参数：
- drift_severity: high / medium / low（漂移强度）
- drift_affects_performance: true / false（漂移是否真的让当前模型变差）
"""
import os

import numpy as np


def make_fake_segment(n_samples=800, n_features=8, seed=42, drift=False,
                      drift_severity="medium", drift_affects_performance=True, name=None):
    """造一段自洽的假数据（特征 + 标签），模拟正常段或漂移段。

    参数:
        n_samples: 样本数。
        n_features: 特征数（建议 >= 4，前两个特征决定标签）。
        seed: 随机种子。
        drift: True 时注入漂移。
        drift_severity: low / medium / high，控制漂移强度。
        drift_affects_performance: True 时漂移会让旧模型性能明显下降；False 时性能基本不变。
        name: 段名。
    返回:
        数据段 dict：name / X / y / features / pred / scores / meta。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    # 标签只依赖前两个特征 f0、f1；噪声控制可分度（避免 accuracy/auc 自相矛盾）
    noise_scale = 0.7
    logit = 1.0 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(0.0, noise_scale, n_samples)

    if drift:
        shift = {"low": 0.4, "medium": 1.0, "high": 2.0}[drift_severity]
        scale = {"low": 1.1, "medium": 1.4, "high": 2.0}[drift_severity]

        if drift_affects_performance:
            # 概念漂移 + 特征漂移：改变标签函数，同时偏移相关特征，让旧模型失效
            X[:, 0] += shift
            X[:, 1] *= scale
            logit = 0.4 * X[:, 0] + 0.9 * X[:, 1] + rng.normal(0.0, noise_scale, n_samples)
        else:
            # 纯数据漂移：只偏移无关特征 f2..f7，标签函数不变 -> 性能基本不变
            if n_features >= 3:
                X[:, 2:] = X[:, 2:] * scale + shift

    y = (logit > 0.0).astype(int)
    return {
        "name": name or ("drift" if drift else "normal"),
        "X": X,
        "y": y,
        "features": [f"f{i}" for i in range(n_features)],
        "pred": None,
        "scores": None,
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "seed": seed,
            "drift": drift,
            "drift_severity": drift_severity,
            "drift_affects_performance": drift_affects_performance,
        },
    }


def load_data(path=None, **kwargs):
    """真实数据加载接口。

    已支持 gas 传感器 batch 文件（.dat，libsvm 格式，练习用）；
    官方数据包接入时在此扩展。返回与 make_fake_segment 相同结构的数据段。
    """
    if path and os.path.isfile(path) and path.lower().endswith(".dat"):
        return load_gas_batch(path, **kwargs)
    raise NotImplementedError("官方数据接口尚未接入：请在 load_data 中读取官方数据包。")


def _parse_libsvm_line(line):
    """# new：解析 libsvm 行 "<label> <idx>:<val> ..."，返回 (label, {idx: val})。"""
    parts = line.split()
    label = int(float(parts[0]))
    feats = {}
    for kv in parts[1:]:
        k, v = kv.split(":")
        feats[int(k)] = float(v)
    return label, feats


def load_gas_batch(path, n_features=128, target_class=1, name=None):
    """# modified：加载 gas 传感器漂移数据集的单个 batch（libsvm 格式）。

    参数:
        path: batch .dat 文件路径。
        n_features: 特征数（gas 数据集为 128）。
        target_class: 二分类目标类别（label==target_class 记 1，其余 0）；
                      传 None 则保留原始多分类标签（1-6，共 6 种气体）。
    返回: 数据段 dict（X / y / features / pred / scores / meta）。
    """
    rows, labels = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, feats = _parse_libsvm_line(line)
            rows.append([feats.get(i, 0.0) for i in range(1, n_features + 1)])
            if target_class is None:
                labels.append(label)                    # 多分类：保留原始类别
            else:
                labels.append(1 if label == target_class else 0)
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    return {
        "name": name or os.path.basename(path),
        "X": X, "y": y,
        "features": [f"f{i}" for i in range(X.shape[1])],
        "pred": None, "scores": None,
        "meta": {"source": path, "n_samples": int(X.shape[0]),
                 "n_features": int(X.shape[1]), "target_class": target_class,
                 "n_classes": 6 if target_class is None else 2},
    }


def load_platform_jsonl(path, task, classes=None, name=None):
    """# new：加载官方训练包 JSONL 为数据段（复用官方 baselines.data 解析器）。

    平台 JSONL 每行 {"id", "inputs", "answer"}，inputs 为嵌套数值结构；
    由官方 read_dataset 展平为特征矩阵（schema 自动校验、禁止目标衍生字段）。
    返回 segment：X（n×d）/ y（分类=类编号、回归=原单位数值）/ meta（paths/sha256/task）。
    """
    from tools.baseline import data as bdata
    ds = bdata.read_dataset(path, task=task, classes=classes)
    return {
        "name": name or os.path.basename(path),
        "X": ds["x"], "y": ds["y"],
        "features": ds["paths"],
        "pred": None, "scores": None,
        "meta": {"source": path, "n_samples": int(ds["x"].shape[0]),
                 "n_features": int(ds["x"].shape[1]), "sha256": ds["sha256"],
                 "task": task, "classes": classes},
    }


def validate_segment(segment, config=None):
    """# new：数据契约校验——字段完整性、类型、缺失率、标签合法性。返回 (是否通过, 原因)。"""
    if not isinstance(segment, dict):
        return False, "数据段必须是 dict"
    for key in ("X", "y"):
        if key not in segment:
            return False, f"数据段缺少字段 [{key}]"
    X = np.asarray(segment["X"], dtype=float)
    y = np.asarray(segment["y"])
    if X.ndim != 2:
        return False, f"X 必须是二维数组，实际 {X.ndim} 维"
    if X.shape[0] == 0:
        return False, "X 样本数为 0"
    if y.shape[0] != X.shape[0]:
        return False, f"X 与 y 样本数不一致（{X.shape[0]} vs {y.shape[0]}）"
    if np.isnan(y).any():
        return False, "y 含有 NaN"
    # 标签必须为非负整数（二分类 0/1；多分类 0..k-1）
    if np.any(np.asarray(y) < 0) or np.any(np.asarray(y) != np.round(np.asarray(y))):
        return False, "y 必须为非负整数标签"
    from core import cfg_get, load_config
    config = config or load_config()
    max_missing = cfg_get(config, "drift.missing_trigger", 0.15)
    if float(np.isnan(X).mean()) > max_missing:
        return False, f"X 缺失率 {np.isnan(X).mean():.3f} 超过上限 {max_missing}"
    return True, "ok"
