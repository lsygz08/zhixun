# 智训引擎系统镜像（离线交付）
# 基础镜像：python:3.10-slim（赛题组指定；具体补丁号构建时解析，verify 脚本第2步确认）
FROM python:3.10-slim

# 非交互式前端（避免 apt 卡在交互提示）
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 中文 locale + 时区（赛题数据可能含中文路径/标签，避免 locale 报错）
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales tzdata \
    && sed -i 's/^# *zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=zh_CN.UTF-8 LANGUAGE=zh_CN:zh LC_ALL=zh_CN.UTF-8 TZ=Asia/Shanghai

# Python 强制 UTF-8 模式（3.10 默认不启用，强制后中文输出/文件名不会在评测环境报错）
ENV PYTHONUTF8=1 PYTHONIOENCODING=utf-8

# 依赖固化进镜像（构建期安装，运行期不安装、不联网）
# pip 源：默认官方 PyPI（评测组可复现构建）；本地构建可传镜像加速：
#   docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple ...
ARG PIP_INDEX=
COPY requirements.txt .
RUN if [ -n "$PIP_INDEX" ]; then pip install --no-cache-dir -i $PIP_INDEX -r requirements.txt;     else pip install --no-cache-dir -r requirements.txt; fi

COPY . .

# 说明：pytrio（PyTRiO 训练平台 SDK）不进镜像——只在开发机做训练任务用；
# 镜像内 llm.enabled=false，大模型走 /opt/models/qwen3.8-27b 本地权重 + 本地推理服务，不需要 pytrio。
# 若坚持要装：pytrio==0.2.9 在 3.10 可装（Requires-Python>=3.10，transformers 自动解析到 4.57.3，5.x 需 3.11+）。

# 默认入口 = 复现脚本（docker run 直接输出 accuracy/f1/auc 三指标）
# 完整闭环另行运行：docker run --rm zhixun:latest python main.py
CMD ["python", "reproduce.py"]
