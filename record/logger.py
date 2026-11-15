"""实验记录模块：落本地 JSONL，预留 SwanLab 接口。

每条记录包含 10 个核心字段：
实验编号 / 触发原因 / 数据快照哈希 / 代码版本 / 配置全量 / 随机种子 /
资源消耗(GPU/标注/Token) / 评估结果 / 决策理由 / 产物路径。
"""
import datetime
import hashlib
import json
import os
import random
import subprocess

from core import PROJECT_ROOT, load_config

_OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
_EXPERIMENTS = os.path.join(_OUTPUTS, "experiments")
_ARTIFACTS = os.path.join(_OUTPUTS, "artifacts")
_PENDING = os.path.join(_EXPERIMENTS, ".pending_queue.json")

_RUN_ID = None
_CONFIG = None
_SEED = None
_SEQ = 0
_CYCLE = 0   # new：当前轮次号，写入 cycle_id


def _gen_run_id():
    """生成 run_id：run_YYYYMMDD_HHMMSS_随机4位。"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices("0123456789abcdef", k=4))
    return f"run_{ts}_{suffix}"


def _config_digest():
    """# new：配置摘要（用于 run_id 的"轮次-动作-配置摘要"）。"""
    try:
        s = json.dumps(_CONFIG, ensure_ascii=False, sort_keys=True, default=_json_default)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return "cfg"


def _code_version():
    """代码版本：优先 git 短 commit，失败则用时间戳。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "ts_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def init(run_id=None, config=None, seed=None):
    """程序启动时初始化：生成 run_id、准备输出目录。返回 run_id。"""
    global _RUN_ID, _CONFIG, _SEED, _SEQ
    _CONFIG = config or load_config()
    _SEED = seed if seed is not None else (_CONFIG.get("run", {}) or {}).get("seed", 42)
    _RUN_ID = run_id or _gen_run_id()
    _SEQ = 0
    os.makedirs(_EXPERIMENTS, exist_ok=True)
    os.makedirs(_ARTIFACTS, exist_ok=True)
    return _RUN_ID


def run_id():
    """返回当前 run_id。"""
    return _RUN_ID


def set_cycle(cycle):
    """# new：设置当前轮次号（写入记录的 cycle_id）。"""
    global _CYCLE
    _CYCLE = cycle


def hash_segment(segment):
    """计算数据段快照哈希（X + y 的 sha256 前 16 位）。"""
    try:
        import numpy as np
        X = np.asarray(segment.get("X", []))
        y = np.asarray(segment.get("y", []))
        h = hashlib.sha256()
        h.update(X.tobytes())
        h.update(b"|")
        h.update(y.tobytes())
        return h.hexdigest()[:16]
    except Exception:
        return "hash_error"


def _json_default(o):
    """JSON 序列化兜底：处理 numpy 类型。"""
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def log(step, detail=None, trigger_reason=None, data_hash=None, resource=None,
        eval_result=None, decision_reason=None, artifact_path=None,
        data_snapshot=None, code=None, metrics=None, conclusion=None):
    """# modified：追加一条记录到 outputs/experiments/<run_id>.jsonl（每行一条）。

    记录结构对齐官方附录 B：run_id / trigger / data_snapshot / code / config /
    seed / cost / metrics / reason / artifacts / conclusion。
    若记录后端失败，走降级路径：落本地待补传队列 .pending_queue.json。
    """
    global _SEQ
    _SEQ += 1
    detail = detail or {}
    trigger = trigger_reason if isinstance(trigger_reason, dict) else (
        {"drift_type": trigger_reason} if trigger_reason else {})
    snap = data_snapshot or ({"hash": data_hash} if data_hash else {})
    code_info = code or {"image_digest": None, "commit": _code_version()}
    record = {
        "run_id": f"{_CYCLE}-{step}-{_config_digest()}",      # 轮次-动作-配置摘要
        "trigger": trigger,                                   # 触发原因
        "data_snapshot": snap,                                # 数据快照
        "code": code_info,                                    # 代码与镜像
        "config": _CONFIG,                                    # 配置全量
        "seed": _SEED,                                        # 随机种子
        "cost": resource if resource is not None else {},     # 资源消耗 gpu/labels/tokens
        "metrics": metrics if metrics is not None else (eval_result if eval_result is not None else {}),
        "reason": decision_reason or None,                    # 决策理由
        "artifacts": [artifact_path] if artifact_path else [],  # 产物路径
        "conclusion": conclusion or None,                     # 结论
        "cycle_id": _CYCLE,                                   # 额外保留：第几轮
        "experiment_id": f"{_RUN_ID}_{step}_{_SEQ:03d}",      # 额外保留：唯一记录号
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "run_id": _RUN_ID,
        "detail": detail,
    }
    line = json.dumps(record, ensure_ascii=False, default=_json_default)
    path = os.path.join(_EXPERIMENTS, f"{_RUN_ID}.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        # 故障降级路径（问题三·5）：记录后端失败 -> 先落本地队列，后续补传
        record["detail"]["error_code"] = "E4001"
        _enqueue_pending(record)
        print(f"[logger] 记录后端写入失败，已入本地待补传队列：{e}")
    return record


def _enqueue_pending(record):
    """把失败记录写入本地待补传队列文件。"""
    try:
        queue = []
        if os.path.exists(_PENDING):
            with open(_PENDING, "r", encoding="utf-8") as f:
                queue = json.load(f)
        queue.append(record)
        with open(_PENDING, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
    except (OSError, ValueError) as e:
        print(f"[logger] 待补传队列写入失败：{e}")


def flush_pending():
    """# modified：把本地待补传队列逐条补传到 SwanLab；成功移除，失败保留原时间戳。"""
    if not os.path.exists(_PENDING):
        return 0
    try:
        with open(_PENDING, "r", encoding="utf-8") as f:
            queue = json.load(f)
    except (OSError, ValueError):
        return 0
    if not queue:
        return 0
    uploaded, still = 0, []
    for rec in queue:
        if upload_swanlab(rec):
            uploaded += 1
        else:
            still.append(rec)
    try:
        if still:
            with open(_PENDING, "w", encoding="utf-8") as f:
                json.dump(still, f, ensure_ascii=False, indent=2)
        else:
            os.remove(_PENDING)
    except OSError as e:
        print(f"[logger] 待补传队列写回失败：{e}")
    print(f"[logger] 补传完成：成功 {uploaded} 条，仍待补传 {len(still)} 条")
    return uploaded


_SWANLAB_INITED = False


def upload_swanlab(record=None):
    """上传单条记录到 SwanLab 云端。成功返回 True，失败返回 False（留在本地队列）。"""
    global _SWANLAB_INITED
    if record is None:
        return False
    try:
        import swanlab
        cfg = load_config().get("record") or {}
        if not _SWANLAB_INITED:
            api_key = cfg.get("swanlab_api_key")
            project = cfg.get("swanlab_project") or "zhixun"
            if api_key:
                swanlab.login(api_key=api_key)
            swanlab.init(project=project)
            _SWANLAB_INITED = True
        # 固定键集合（避免网页出现随记录类型漂移的稀疏曲线）
        m = (record.get("metrics") or {}).get("valid") or (record.get("metrics") or {})
        log_data = {}
        for k in ("accuracy", "f1", "auc", "rmse", "mae", "r2"):
            if k in m and isinstance(m[k], (int, float)) and not isinstance(m[k], bool):
                log_data[k] = float(m[k])
        used = (record.get("cost") or {}).get("used") or {}
        for k in ("gpu_hours", "labels", "tokens"):
            if k in used and isinstance(used[k], (int, float)):
                log_data["cost_" + k] = float(used[k])
        log_data["cycle_id"] = record.get("cycle_id") or 0
        swanlab.log(log_data)
        return True
    except Exception as e:
        print(f"[logger] SwanLab 上传失败：{e}")
        return False


def sync_run(rid=None):
    """把某次运行的本地记录逐条同步到 SwanLab。返回成功条数。"""
    path = experiments_path(rid)
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if upload_swanlab(rec):
                n += 1
    except OSError:
        pass
    print(f"[logger] SwanLab 同步：成功 {n} 条")
    return n


def experiments_dir():
    """返回 experiments 目录。"""
    return _EXPERIMENTS


def artifacts_dir():
    """返回 artifacts 目录。"""
    return _ARTIFACTS


def experiments_path(rid=None):
    """返回当前 run 的 JSONL 记录文件路径。"""
    return os.path.join(_EXPERIMENTS, f"{rid or _RUN_ID}.jsonl")
