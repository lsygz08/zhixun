"""复现测试：同种子重建初始模型，指标确定且合法。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import reproduce


def main():
    ok1, m1, _, _ = reproduce.reproduce_static_baseline(tolerance=0.01)
    ok2, m2, _, _ = reproduce.reproduce_static_baseline(tolerance=0.01)
    assert ok1 and ok2
    assert m1 == m2, (m1, m2)
    assert 0.5 < m1["accuracy"] < 1.0, m1
    assert 0.5 < m1["f1"] < 1.0 and 0.5 < m1["auc"] <= 1.0, m1
    print("test_reproduce PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
