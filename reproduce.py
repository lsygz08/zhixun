"""复现静态基线（7.3 第4项自检）：重建初始模型并验证可复现。

运行：python reproduce.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import reproduce

if __name__ == "__main__":
    ok, metrics, recorded, dev = reproduce.reproduce_static_baseline()
    print(f"复现指标: {metrics}")
    if recorded is None:
        print("记录指标: 无历史记录可比对（首次运行）")
        print("主指标偏差: N/A（无对比对象）")
        print("结论: 复现计算完成（固定种子，确定性）")
    else:
        print(f"记录指标: {recorded}")
        print(f"主指标偏差: {dev:.6f}")
        print("结论:", "可复现" if ok else "不可复现")
