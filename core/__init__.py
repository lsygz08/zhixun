"""core 包：核心闭环环节（监测 / 诊断 / 优化 / 门禁 / 回滚）。

本包还提供共享工具：
- load_config()：读取项目根目录的 config.yaml，带缓存。
- cfg_get()：按点号路径安全取值，避免 KeyError。

所有模块从这里读取配置，代码里不硬编码魔法数字。
"""
import os

# 项目根目录 = core 的上一级（即 zhixun/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

_CONFIG_CACHE = None


def _strip_inline_comment(s):
    """去掉行内注释（形如 ' #...' 且不在引号内）。"""
    in_quote = None
    for i, ch in enumerate(s):
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None and i > 0 and s[i - 1] == " ":
            return s[:i].rstrip()
    return s


def _parse_scalar(s):
    """把标量字符串转成 Python 对象（支持 bool/int/float/字符串/内联列表）。"""
    s = s.strip()
    if s in ("", "null", "~", "None"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    try:
        if any(c in s for c in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _tokenize(text):
    """把 YAML 文本切成 (缩进, 内容) 列表，去掉空行与注释。"""
    tokens = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = _strip_inline_comment(line.strip())
        if content:
            tokens.append((indent, content))
    return tokens


def _parse_mapping(tokens, i, indent):
    """递归解析一个映射（key: value）。返回 (dict, 下一个下标)。"""
    result = {}
    n = len(tokens)
    while i < n:
        ti, content = tokens[i]
        if ti < indent:
            break
        if ti > indent or content.startswith("- ") or ":" not in content:
            i += 1
            continue
        key, val = content.split(":", 1)
        key = key.strip().strip("'\"")
        val = val.strip()
        if val == "":
            if i + 1 < n and tokens[i + 1][0] > indent:
                nxt_indent, nxt_content = tokens[i + 1]
                if nxt_content.startswith("- "):
                    lst, i = _parse_list(tokens, i + 1, nxt_indent)
                    result[key] = lst
                else:
                    nested, i = _parse_mapping(tokens, i + 1, nxt_indent)
                    result[key] = nested
            else:
                result[key] = None
                i += 1
        else:
            result[key] = _parse_scalar(val)
            i += 1
    return result, i


def _parse_list(tokens, i, indent):
    """递归解析一个列表（- item）。返回 (list, 下一个下标)。"""
    result = []
    n = len(tokens)
    while i < n:
        ti, content = tokens[i]
        if ti < indent:
            break
        if content.startswith("- "):
            result.append(_parse_scalar(content[2:].strip()))
        i += 1
    return result, i


def _load_yaml_minimal(text):
    """内置最小 YAML 解析器：未安装 pyyaml 时兜底。"""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    result, _ = _parse_mapping(tokens, 0, tokens[0][0])
    return result


def load_config(path=None, force=False):
    """读取 config.yaml。优先用 pyyaml，未安装时用内置最小解析器。带缓存。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force:
        return _CONFIG_CACHE
    p = path or CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    data = None
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        data = _load_yaml_minimal(text)
    if not isinstance(data, dict):
        data = {}
    _CONFIG_CACHE = data
    return data


def cfg_get(config, dotted, default=None):
    """按点号路径取值，例如 cfg_get(config, 'drift.psi_warn')。"""
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
