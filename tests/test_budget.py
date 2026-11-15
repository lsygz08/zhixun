"""测试 4：预算扣费合理。"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _util import run_main, patch_config, restore_config


def test_unit():
    from core import load_config, optimize
    from budget.tracker import BudgetTracker
    config = load_config()
    b = BudgetTracker(config.get("budget", {}))
    cost = optimize.action_cost("resample_or_calibrate", config)
    assert cost.get("labels") == 200, f"resample 每次应扣 200，实际 {cost}"
    b.spend(cost)
    b.spend(cost)
    assert b.summary()["used"]["labels"] == 400, "两次应累计 400（累加）"
    assert b.summary()["remaining"]["labels"] > 0, "labels 剩余应 > 0"
    print(f"  PASS：resample 每次扣 200，2 次累计 400，剩余 {b.summary()['remaining']['labels']}")


def test_integration():
    original = patch_config({"max_rounds": 3})
    try:
        rc, text = run_main()
        assert rc == 0, f"main.py 退出码 {rc}"
        lines = [l for l in text.splitlines() if "[budget]" in l and "扣费" in l]
        assert lines, "应有扣费日志"
        remainings = []
        for l in lines:
            m = re.search(r"剩余额度 (\{.*?\})", l)
            assert m, f"扣费行应含剩余额度：{l}"
            remainings.append(ast.literal_eval(m.group(1)))
        for key in ("gpu_hours", "labels", "tokens"):
            seq = [r[key] for r in remainings]
            assert all(v >= 0 for v in seq), f"{key} 剩余应非负：{seq}"
            assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)), f"{key} 剩余应非增：{seq}"
        assert remainings[-1]["labels"] > 0, "labels 剩余不能为 0"
        print(f"  PASS：扣费递减非负，labels 剩余 {remainings[-1]['labels']}")
    finally:
        restore_config(original)


if __name__ == "__main__":
    test_unit()
    test_integration()
    print("test_budget PASS")
