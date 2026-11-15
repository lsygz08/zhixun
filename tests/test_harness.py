"""Harness 测试：白名单校验 / 输出解析 / 提示词 / 未启用不决策。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import harness


def main():
    ok, _ = harness.validate_action({"name": "finetune"})
    assert ok
    ok, reason = harness.validate_action({"name": "delete_weights"})
    assert not ok and "白名单" in reason, reason
    ok, reason = harness.validate_action({"x": 1})
    assert not ok and "name" in reason, reason

    assert harness.parse_action('{"name": "finetune", "params": {}}')["name"] == "finetune"
    assert harness.parse_action('说明文字 {"name": "no_action"} 结尾')["name"] == "no_action"
    assert harness.parse_action("乱码") is None
    assert harness.parse_action("") is None
    assert harness.parse_action(None) is None

    p = harness.build_prompt({"card": {"drift_type": "concept_drift"}}, {"gpu_hours": 47.8})
    assert "动作清单" in p and "concept_drift" in p and "47.8" in p, p

    cfg = {"llm": {"enabled": False}}
    assert harness.decide_plan({"drift_type": "concept_drift"}, {}, config=cfg) is None
    print("test_harness PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
