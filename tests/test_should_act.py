"""测试 1：should_act 的"动/不动"判断。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _util import run_main, patch_config, restore_config


def check_A():
    """drift_affects_performance=False：漂移严重但性能不掉 -> 应不动。"""
    original = patch_config({"drift_affects_performance": "false", "max_rounds": 3})
    try:
        rc, text = run_main()
        assert rc == 0, f"A 场景 main.py 退出码 {rc}"
        assert "性能未退化" in text, "A：应输出'性能未退化'"
        assert "无需动作" in text, "A：应输出'无需动作'"
        assert "[finetune]" not in text, "A：不应执行 finetune"
        assert "重采样 + Platt" not in text, "A：不应执行 resample"
        print("  A PASS：should_act 判不动，未执行任何动作")
    finally:
        restore_config(original)


def check_B():
    """drift_affects_performance=True：漂移且性能掉 -> 应要动并执行动作。"""
    original = patch_config({"drift_affects_performance": "true", "max_rounds": 3})
    try:
        rc, text = run_main()
        assert rc == 0, f"B 场景 main.py 退出码 {rc}"
        assert "漂移触发且当前模型性能退化" in text, "B：should_act 应判要动"
        acted = ("[finetune]" in text) or ("重采样 + Platt" in text)
        assert acted, "B：应执行了对应动作"
        print("  B PASS：should_act 判要动，执行了动作")
    finally:
        restore_config(original)


if __name__ == "__main__":
    check_A()
    check_B()
    print("test_should_act PASS")
