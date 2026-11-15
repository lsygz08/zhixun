"""测试 3：三条故障降级开关。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _util import run_main, patch_config, restore_config


def run_all():
    cases = [
        ("finetune 失败", "inject_finetune_failure", "降档重试"),
        ("diagnose 失败", "inject_diagnose_failure", "切纯规则策略"),
        ("gate 失败", "inject_gate_failure", "门禁异常已降级"),
    ]
    for name, key, keyword in cases:
        original = patch_config({key: "true", "max_rounds": 2})
        try:
            rc, text = run_main()
            assert rc == 0, f"{name}：main.py 退出码 {rc}（应不崩溃）"
            assert keyword in text, f"{name}：日志应含 '{keyword}'"
            print(f"  PASS：{name}（returncode=0，含 '{keyword}'）")
        finally:
            restore_config(original)


if __name__ == "__main__":
    run_all()
    print("test_fault_injection PASS")
