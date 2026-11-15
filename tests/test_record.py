"""测试 5：实验记录 10 字段齐全。"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _util import run_main, latest_jsonl

REQUIRED = [
    "run_id", "trigger", "data_snapshot", "code", "config", "seed",
    "cost", "metrics", "reason", "artifacts", "conclusion",
]


def test():
    rc, text = run_main()
    assert rc == 0, f"main.py 退出码 {rc}"
    # 优先从运行摘要里解析记录文件路径
    m = re.search(r"实验记录：(.+\.jsonl)", text)
    path = m.group(1).strip() if m else latest_jsonl()
    assert path, "未找到 .jsonl 文件"
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert records, "jsonl 为空"
    for rec in records:
        missing = [k for k in REQUIRED if k not in rec]
        assert not missing, f"记录缺字段 {missing}"
    print(f"  PASS：{len(records)} 条记录，10 字段齐全（文件 {Path(path).name}）")


if __name__ == "__main__":
    test()
    print("test_record PASS")
