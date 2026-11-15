"""策略层自改进：策略记忆（两层自进化中的策略层）。

从历史实验记录（outputs/experiments/*.jsonl）学习：
- 每种 (漂移类型, 建议动作) 组合历史上试过多少次、成功上线多少次；
- 优化器决策时优先选择"历史成功率更高"的动作，实现策略层自改进。
"""
import glob
import json
import os


def load_memory(experiments_dir):
    """扫描历史 JSONL，返回 {(drift_type, action): {"success": n, "total": n}}。"""
    memory = {}
    for path in sorted(glob.glob(os.path.join(experiments_dir, "*.jsonl"))):
        cards = {}      # cycle_id -> 归因卡
        promoted = {}   # cycle_id -> 是否上线
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    cid = rec.get("cycle_id")
                    step = rec.get("step")
                    if step == "diagnose":
                        card = (rec.get("detail") or {}).get("card") or {}
                        if card.get("drift_type") and card.get("suggested_action"):
                            cards[cid] = card
                    elif step == "rollback":
                        detail = rec.get("detail") or {}
                        promoted[cid] = detail.get("promoted") is not None
        except OSError:
            continue
        for cid, card in cards.items():
            key = (card["drift_type"], card["suggested_action"])
            m = memory.setdefault(key, {"success": 0, "total": 0})
            m["total"] += 1
            if promoted.get(cid):
                m["success"] += 1
    return memory


def suggest_action(memory, drift_type, default_action):
    """基于策略记忆，为某漂移类型挑选历史成功率最高的动作；无记忆则返回默认。

    返回 (动作, 历史成功率)；无记忆时成功率返回 -1。
    """
    best_action = default_action
    best_rate = -1.0
    for (dt, action), m in memory.items():
        if dt == drift_type and m["total"] > 0:
            rate = m["success"] / m["total"]
            if rate > best_rate:
                best_action = action
                best_rate = rate
    return best_action, best_rate
