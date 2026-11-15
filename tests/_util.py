"""测试辅助函数：配置修改、运行 main.py、读取最新 jsonl。"""
import glob
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # zhixun/


def _env():
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"            # 保证 main.py 输出 UTF-8，避免中文乱码
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def set_key(text, key, value):
    """把 config 文本里某 key 的值改成 value，保留行内注释。"""
    pattern = rf"^(\s*{re.escape(key)}:\s*)\S+(\s*#.*)?$"
    return re.sub(pattern, lambda m: f"{m.group(1)}{value}{m.group(2) or ''}", text, flags=re.M)


def patch_config(changes):
    """changes: {key: value}，写入 config.yaml，返回原始文本（用于 restore）。"""
    cfg = ROOT / "config.yaml"
    original = cfg.read_text(encoding="utf-8")
    text = original
    for key, value in changes.items():
        text = set_key(text, key, value)
    cfg.write_text(text, encoding="utf-8")
    return original


def restore_config(original):
    (ROOT / "config.yaml").write_text(original, encoding="utf-8")


def run_main():
    """运行 main.py，stdout/stderr 重定向到临时文件，返回 (returncode, 输出文本)。"""
    fd, logpath = tempfile.mkstemp(suffix=".log", prefix="zhixun_")
    os.close(fd)
    with open(logpath, "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [sys.executable, str(ROOT / "main.py")],
            cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=300, env=_env(),
        ).returncode
    text = open(logpath, encoding="utf-8").read()
    os.unlink(logpath)
    return rc, text


def latest_jsonl():
    """返回 outputs/experiments/ 下最新（mtime）的 .jsonl 路径，无则返回 None。"""
    files = glob.glob(str(ROOT / "outputs" / "experiments" / "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None

def run_platform():
    """运行平台闭环入口（main_platform.py，A 包 3 轮），返回 (退出码, 输出文本)。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "main_platform.py",
           "--pack", os.path.join("data", "official", "cpu-a-guide-20261110-v1"),
           "--profile", "A", "--rounds", "3", "--epochs", "1",
           "--output", os.path.join("outputs", "platform", "A")]
    cp = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", env=_env())
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
