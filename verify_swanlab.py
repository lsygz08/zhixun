"""验证 SwanLab 云端连接（pip install swanlab 后运行一次）。

运行：python verify_swanlab.py
成功后到你的 SwanLab 网页「智造进化论」项目里能看到一条 test_acc 记录。
"""
import sys

try:
    import swanlab
except ImportError:
    print("未安装 swanlab，请先执行：pip install swanlab")
    sys.exit(1)

from core import load_config

cfg = load_config().get("record") or {}
api_key = cfg.get("swanlab_api_key")
project = cfg.get("swanlab_project") or "智造进化论"

if api_key:
    swanlab.login(api_key=api_key)
swanlab.init(project=project)
swanlab.log({"test_acc": 0.99, "cycle_id": 0})
print("上传成功，请到 SwanLab 网页查看")
