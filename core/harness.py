"""Harness 接口（官方第 4 章）：大模型只判断，动作由确定性工具执行。

驱动模型：PyTRiO 的 Qwen3.8-27B（本地推理，离线，OpenAI 兼容接口）。
默认"先规则后智能"：llm.enabled=false 或大模型不可用时，回退规则决策。
"""
import json
import re

from core import cfg_get, load_config

ACTION_WHITELIST = ("resample_or_calibrate", "finetune", "no_action")

_ROLE_CONSTRAINTS = (
    "你是工业模型自进化系统的决策核心。只能从给定动作清单中选择一个动作，"
    "输出 JSON 对象：{\"name\": 动作名, \"params\": {...}, "
    "\"expected_effect\": \"预期效果\", \"reason\": \"理由\"}。"
    "不得输出清单之外的动作；不得修改评测脚本、数据口径或指标定义；"
    "不得访问外网或读取全量日志。"
)

_ACTION_LIST = (
    "动作清单："
    "1. resample_or_calibrate：重采样+概率校准（数据漂移用）；"
    "2. finetune：增量微调（概念漂移用）；"
    "3. no_action：本轮不改动模型。"
)


def build_prompt(evidence, budget_summary=None):
    """构造决策提示词：角色硬约束 + 动作清单 + 证据摘要 + 预算余额。"""
    parts = [_ROLE_CONSTRAINTS, _ACTION_LIST]
    parts.append("证据摘要：" + json.dumps(evidence, ensure_ascii=False, default=str)[:2000])
    if budget_summary:
        parts.append("预算余额：" + json.dumps(budget_summary, ensure_ascii=False, default=str))
    parts.append("请只输出 JSON，不要多余文字。")
    return "\n".join(parts)


def call_llm_pytrio(prompt, config=None):
    """# new：调用 PyTRiO 官方基础模型（pytrio SDK）。任何失败返回 None（回退规则）。

    实测确认（2026-11-15）：SDK 正常、模型列表含 Qwen/Qwen3.8-27B；
    额度未发放时平台返回 billing insufficient balance，此处捕获后走规则回退。
    """
    config = config or load_config()
    llm_cfg = config.get("llm") or {}
    model = llm_cfg.get("model", "Qwen/Qwen3.8-27B")
    try:
        from pytrio import ServiceClient, ModelInput, SamplingParams
        from pytrio.types.forward.encoded_text_chunk import EncodedTextChunk
        sc = ServiceClient()
        sampling = sc.create_sampling_client(model)
        tokenizer = sampling.get_tokenizer()
        encode = getattr(tokenizer, "encode", None) or getattr(tokenizer, "tokenize", None)
        if encode is None:
            raise RuntimeError("tokenizer 无 encode 方法")
        tokens = encode(prompt)
        mi = ModelInput(chunks=[EncodedTextChunk(tokens=list(tokens))])
        fut = sampling.sample(mi, num_samples=1,
                              sampling_params=SamplingParams(max_tokens=512, temperature=0.2),
                              return_text=True)
        resp = fut.result() if hasattr(fut, "result") else fut
        seq = resp.sequences[0] if getattr(resp, "sequences", None) else None
        return getattr(seq, "text", None)
    except Exception as e:
        print(f"[harness] PyTRiO 调用失败（回退规则）：{e}")
        return None


def call_llm(prompt, config=None):
    """调用大模型：优先 PyTRiO SDK，失败回退本地 OpenAI 兼容接口，再失败返回 None。"""
    config = config or load_config()
    llm_cfg = config.get("llm") or {}
    if llm_cfg.get("backend", "local") == "pytrio":
        text = call_llm_pytrio(prompt, config)
        if text:
            return text
        print("[harness] 回退本地推理接口…")
    base_url = llm_cfg.get("base_url", "http://localhost:8000/v1")
    model = llm_cfg.get("model", "qwen3.8-27b")
    try:
        import urllib.request
        payload = json.dumps({"model": model,
                              "messages": [{"role": "user", "content": prompt}],
                              "temperature": 0.2, "max_tokens": 512}).encode("utf-8")
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[harness] 大模型调用失败：{e}")
        return None


def parse_action(text):
    """从大模型回复中解析动作 JSON。失败返回 None。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate_action(action):
    """校验动作对象 schema 与白名单。返回 (是否通过, 原因)。"""
    if not isinstance(action, dict):
        return False, "动作必须是字典对象"
    if "name" not in action:
        return False, "动作缺少 name 字段"
    if action["name"] not in ACTION_WHITELIST:
        return False, f"动作 {action['name']!r} 不在白名单内"
    return True, "ok"


def llm_decide(evidence, config=None, budget_summary=None):
    """调用大模型输出动作对象；任何失败返回 None（调用方回退规则）。"""
    config = config or load_config()
    text = call_llm(build_prompt(evidence, budget_summary), config)
    action = parse_action(text)
    ok, reason = validate_action(action) if action else (False, "无输出")
    if not ok:
        print(f"[harness] 大模型动作校验失败：{reason}，回退规则决策")
        return None
    return action


def decide_plan(card, stats, config=None, budget=None):
    """尝试用大模型决策生成 plan；未启用/失败返回 None（调用方回退规则）。"""
    config = config or load_config()
    if not cfg_get(config, "llm.enabled", False):
        return None
    action = llm_decide({"card": card, "stats": stats}, config,
                        budget.summary() if budget is not None else None)
    if action is None:
        return None
    return {
        "action": action["name"],
        "params": action.get("params") or {},
        "reason": f"大模型决策：{action.get('reason', '')}（预期：{action.get('expected_effect', '')}）",
        "expected_gain": float(card.get("expected_gain", 0.01)),
        "card": card,
        "llm": True,
    }
