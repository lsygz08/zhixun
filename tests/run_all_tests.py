"""依次运行 5 个测试，打印汇总表。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = [
    "test_should_act.py",
    "test_gate.py",
    "test_fault_injection.py",
    "test_budget.py",
    "test_record.py",
    "test_strategy.py",
    "test_harness.py",
    "test_reproduce.py",
]


def _env():
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def main():
    results = []
    for t in TESTS:
        print(f"\n==== 运行 {t} ====")
        cp = subprocess.run([sys.executable, str(ROOT / t)], cwd=str(ROOT), env=_env())
        ok = cp.returncode == 0
        results.append((t, ok))

    print("\n" + "=" * 44)
    print("TEST SUMMARY")
    print("=" * 44)
    for t, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {t}")
    total = sum(1 for _, ok in results if ok)
    print("-" * 44)
    print(f"TOTAL: {total}/{len(results)} passed")
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
