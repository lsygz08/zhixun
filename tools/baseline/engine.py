"""官方基线续训引擎：加载检查点 → 续训 → 保存 → 官方口径指标。

续训语义（官方红线）：只 fit 追加轮次，保留权重、Adam 状态、标准化与目标缩放；
禁止重新初始化。每轮保存新 checkpoint 目录（不覆盖上一轮）。
"""
import json
import os

import numpy as np

from tools.baseline import data as bdata
from tools.baseline.model import Model


def load_profile(pack_root, profile):
    """加载 profile 配置 + 基线检查点 + 公开训练/验证集。"""
    cfg = json.load(open(os.path.join(pack_root, "baselines", "configs", profile + ".json"),
                         encoding="utf-8"))
    model = Model.load(os.path.join(pack_root, "checkpoint.npz"))
    train = bdata.read_dataset(os.path.join(pack_root, "public", "train.jsonl"),
                               task=cfg["task"], classes=cfg.get("classes"))
    val = bdata.read_dataset(os.path.join(pack_root, "public", "validation.jsonl"),
                             task=cfg["task"], classes=cfg.get("classes"))
    return model, cfg, train, val


def platform_metrics(model, data):
    """官方口径指标（与官方 baselines.train.metrics 完全一致）。"""
    predicted = model.predict(data["x"])
    expected = data["y"]
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == c) & (expected == c)) /
              (np.sum(labels == c) + np.sum(expected == c)) for c in observed]
        return {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
    error = predicted - expected
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def continue_round(model, train, epochs, learning_rate=None):
    """续训一轮：追加 epochs，保留 Adam/标准化状态。返回逐 epoch loss。"""
    if learning_rate is not None:
        model.meta["learning_rate"] = float(learning_rate)
    return model.fit(train["x"], train["y"], int(epochs))


def run_rounds(pack_root, profile, rounds, epochs, output_root, learning_rates=None):
    """从基线检查点续训多轮。返回 [(阶段, 验证指标, 检查点路径), ...]。"""
    model, cfg, train, val = load_profile(pack_root, profile)
    os.makedirs(output_root, exist_ok=True)
    results = [("baseline", platform_metrics(model, val), None)]
    for r in range(1, int(rounds) + 1):
        lr = (learning_rates[r - 1] if learning_rates and r - 1 < len(learning_rates)
              else cfg.get("learning_rate", 0.003))
        continue_round(model, train, int(epochs), lr)
        m = platform_metrics(model, val)
        out_dir = os.path.join(output_root, f"round-{r}")
        os.makedirs(out_dir, exist_ok=True)
        ckpt = os.path.join(out_dir, "checkpoint.npz")
        model.save(ckpt)
        results.append((f"round-{r}", m, ckpt))
    return results
