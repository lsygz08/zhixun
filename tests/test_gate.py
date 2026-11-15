"""测试 2：门禁三层检查 + 多指标一致性 warning（对齐官方 3.6）。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from core import gate, load_config
from budget.tracker import BudgetTracker


def test():
    config = load_config()
    budget = BudgetTracker(config.get("budget", {}))
    baseline = {"metric": {"accuracy": 0.80, "f1": 0.80, "auc": 0.90},
                "baseline_metrics": {"accuracy": 0.80}}   # 静态基线参照

    # 场景1：绝对指标不达标 -> 第一层拒
    bad = {"metric": {"accuracy": 0.70, "f1": 0.70, "auc": 0.70}}
    r1 = gate.evaluate(bad, baseline, budget, config=config)
    assert r1["pass"] is False, "绝对指标低于线上应拒"
    assert not r1["checks"]["metric_not_worse_than_baseline"]
    print(f"  PASS：第一层拦截（绝对指标低于线上）-> {r1['reason']}")

    # 场景2：指标方向不一致（acc涨但f1/auc跌）-> 降级为 warning，不拒
    conflict = {"metric": {"accuracy": 0.88, "f1": 0.20, "auc": 0.70}}
    r2 = gate.evaluate(conflict, baseline, budget, config=config)
    assert r2["pass"] is True, "多指标一致性已降级为 warning，不应作为拒判条件"
    assert any("指标方向不一致" in w for w in r2["warnings"]), "应产出指标方向不一致 warning"
    print(f"  PASS：多指标冲突降级为 warning -> {r2['warnings']}")

    # 场景3：抗漂移能力不达标 -> 第二层拒（需要漂移回放集）
    seg = {"X": np.zeros((100, 8), dtype=float), "y": np.ones(100, dtype=int)}
    cand = {"weights": [0.0] * 8, "bias": 0.0,
            "metric": {"accuracy": 0.80}}   # 候选在验证集上达标，但在回放集上全错
    r3 = gate.evaluate(cand, baseline, budget, segment=seg, config=config)
    assert r3["pass"] is False, "回放集表现低于静态基线应拒"
    assert not r3["checks"]["anti_drift_not_worse_than_static_baseline"]
    print(f"  PASS：第二层拦截（抗漂移低于静态基线）-> {r3['reason']}")


if __name__ == "__main__":
    test()
    print("test_gate PASS")
