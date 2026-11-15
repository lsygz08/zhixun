"""预算账本模块：记录 GPU 时长 / 标注条数 / Token 三类消耗。

# modified：预算上限读取对齐官方 budget.max_gpu_hours / max_labels / max_tokens。
"""


class BudgetTracker:
    """三类消耗的账本：gpu_hours / labels / tokens。"""

    def __init__(self, limits=None):
        raw = dict(limits or {})
        # 兼容官方 max_* 命名，也兼容旧的 gpu_hours / labels / tokens
        self.limits = {}
        for k in ("gpu_hours", "labels", "tokens"):
            self.limits[k] = float(raw.get("max_" + k, raw.get(k, 0)) or 0)
        self.used = {"gpu_hours": 0.0, "labels": 0, "tokens": 0}

    def check(self, cost=None):
        """判断某次动作是否在预算内（used + cost <= limits）。返回布尔值。"""
        cost = cost or {}
        for key in ("gpu_hours", "labels", "tokens"):
            need = float(cost.get(key, 0) or 0)
            limit = float(self.limits.get(key, 0) or 0)
            if self.used[key] + need > limit:
                return False
        return True

    def spend(self, cost=None):
        """记账：按 cost 逐项累加（一次只扣对应项，不会一次扣满预算）。

        resample 只扣 labels=resample_labels；finetune 只扣 gpu/tokens；llm 只扣 tokens。
        预算不足时拒绝并返回 False，否则返回 True。
        """
        if not self.check(cost):
            return False
        cost = cost or {}
        for key in ("gpu_hours", "labels", "tokens"):
            v = float(cost.get(key, 0) or 0)
            if key in ("labels", "tokens"):
                self.used[key] += int(v)   # 整数额度保持整数
            else:
                self.used[key] += v
        return True

    def summary(self):
        """返回消耗汇总：used / limits / remaining。"""
        remaining = {}
        for key in ("gpu_hours", "labels", "tokens"):
            remaining[key] = float(self.limits.get(key, 0) or 0) - self.used[key]
        return {
            "used": dict(self.used),
            "limits": dict(self.limits),
            "remaining": remaining,
        }
