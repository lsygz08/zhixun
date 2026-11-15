"""策略记忆测试：历史 (漂移类型, 动作) -> 成功率统计与动作优选。"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import strategy


def main():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_strategy")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    try:
        recs = []
        def add(cid, dtype, action, promoted):
            recs.append({"cycle_id": cid, "step": "diagnose",
                         "detail": {"card": {"drift_type": dtype, "suggested_action": action}}})
            recs.append({"cycle_id": cid, "step": "rollback",
                         "detail": {"promoted": "v1" if promoted else None}})
        add(1, "concept_drift", "finetune", True)
        add(2, "concept_drift", "finetune", False)
        add(3, "data_drift", "resample_or_calibrate", True)
        add(4, "data_drift", "resample_or_calibrate", True)
        p = os.path.join(d, "r.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        mem = strategy.load_memory(d)
        assert mem[("concept_drift", "finetune")]["total"] == 2, mem
        assert mem[("concept_drift", "finetune")]["success"] == 1, mem
        assert mem[("data_drift", "resample_or_calibrate")]["success"] == 2, mem

        a, rate = strategy.suggest_action(mem, "unknown_type", "no_action")
        assert a == "no_action" and rate == -1, (a, rate)
        a, rate = strategy.suggest_action(mem, "data_drift", "no_action")
        assert a == "resample_or_calibrate" and rate == 1.0, (a, rate)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("test_strategy PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
