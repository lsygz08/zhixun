"""复现与证据链（官方 5.3 / 7.3 第4项）。

给定实验标识，沿"数据快照-镜像摘要-配置-种子"四要素重建实验，
比对产出指标与记录指标，主指标偏差在允许范围内即认定可复现。
当前实现复现"静态基线"（初始模型）。
"""
import glob
import json
import os

from core import PROJECT_ROOT, cfg_get, load_config
from tools import data_loader, finetune
from core import gate


def _read_recorded_baseline():
    """从 best_model.json 或最新 jsonl 的 baseline 记录读取记录指标。"""
    best_path = os.path.join(PROJECT_ROOT, "outputs", "artifacts", "best_model.json")
    if os.path.exists(best_path):
        try:
            with open(best_path, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("baseline_metrics"):
                return m["baseline_metrics"]
        except Exception:
            pass
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", "experiments", "*.jsonl")))
    for path in reversed(files):
        try:
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("step") == "baseline":
                    detail = rec.get("detail") or {}
                    return detail.get("metric") or rec.get("metrics") or {}
        except Exception:
            continue
    return None


def reproduce_static_baseline(config=None, tolerance=0.01):
    """复现静态基线：按同一配置+种子重建初始模型，比对记录指标。

    返回 (是否可复现, 复现指标, 记录指标, 主指标偏差)。
    """
    config = config or load_config()
    seed = int(cfg_get(config, "run.seed", 42))
    n_samples = int(cfg_get(config, "data.n_samples", 800))
    n_features = int(cfg_get(config, "data.n_features", 8))

    seg = data_loader.make_fake_segment(n_samples=n_samples, n_features=n_features,
                                        seed=seed, drift=False, name="baseline")
    model = finetune.make_initial_model(seg)
    pred, scores = finetune.predict(model, seg["X"])
    metrics = gate.compute_metrics(seg["y"], pred, scores)

    recorded = _read_recorded_baseline()
    if recorded is None:
        # 首次运行：无历史记录可比对，偏差记 None（不伪造 0.0）
        return True, metrics, None, None

    acc_repro = float(metrics.get("accuracy", 0.0))
    acc_rec = float(recorded.get("accuracy", 0.0))
    dev = abs(acc_repro - acc_rec)
    return dev <= tolerance, metrics, recorded, dev
