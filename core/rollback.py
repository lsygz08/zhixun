"""上线 / 回滚模块：版本切换，保留当前最优版本引用。"""
import json
import os


_BEST = None       # 当前最优版本引用
_HISTORY = []      # 历史版本列表


def register_best(model):
    """注册初始最优版本。"""
    global _BEST
    _BEST = model
    _HISTORY.append(model)
    return model


def promote(candidate):
    """上线候选版本：更新当前最优引用并留痕。继承固定基线参照。"""
    global _BEST
    # 基线参照（baseline_metrics）固定不变，跨版本继承，保证 should_act 用统一基准
    if _BEST is not None and "baseline_metrics" in _BEST and "baseline_metrics" not in candidate:
        candidate["baseline_metrics"] = _BEST["baseline_metrics"]
    _BEST = candidate
    _HISTORY.append(candidate)
    return candidate


def rollback():
    """回滚：不采用候选，保持当前最优版本不变，返回最优版本。"""
    return _BEST


def best():
    """返回当前最优版本。"""
    return _BEST


def history():
    """返回历史版本列表（副本）。"""
    return list(_HISTORY)


def reset():
    """清空状态（每次新的 run 调用一次）。"""
    global _BEST, _HISTORY
    _BEST = None
    _HISTORY = []


def persist(artifacts_dir):
    """把当前最优版本持久化到 artifacts/best_model.json，返回路径。"""
    if _BEST is None:
        return None
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, "best_model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_BEST, f, ensure_ascii=False, indent=2)
    return path
