# 版本说明（VERSIONS）

每次修改都在此追加一个新版本条目；本文件是接续工作的唯一入口，先读它。

## 当前状态（v1.9）

- 官方训练包已接入并验证：data/official/（A 气体六类分类 / B 轴承RUL分钟 / C 发动机RUL循环）
- tools/baseline/（官方 NumPy MLP + 续训代码原样复用）+ data_loader.load_platform_jsonl + gate.platform_metrics
- 三包基线检查点复算指标与训练包记录逐位一致（A macro_f1 0.71403、B rmse 681.16、C rmse 52.87）
- 待接：模型层切换为基线 MLP 续训（现 finetune 仍是线性模拟）+ profile 配置 + 镜像入口按平台契约
- 平台：https://rsi-index.115.zone（提交方案→训练迭代→独立评测；试运行不计分）

---

## 版本历史

### v1.11 — 平台闭环 + 提交材料（2026-11-15）
- main_platform.py：10 轮闭环（读历史→诊断→决策→续训→门禁→回滚→记录），大模型决策钩子（llm.enabled 回退规则）、训练时长预算（max_training_seconds）
- 三包闭环实测：A 0.7140→0.7264、B 681.2→647.2、C 52.87→51.85，全部优于基线
- config 加 platform 段（rounds 10 / epochs_per_round / max_training_seconds 1800）
- docs/：技术报告（草稿）、来源声明、演示视频脚本

### v1.10 — 续训引擎（2026-11-15）
- tools/baseline/engine.py：load_profile / continue_round / run_rounds（保留 Adam+标准化状态，官方续训语义）
- 三包续训验证：A macro_f1 0.7140→0.7187；B rmse 681.2→647.2（round-3 劣化 691.9，门禁需拦）；C rmse 52.87→51.93
- 训练速度：A 0.5s / B 1.0s / C 12s（小轮次参考）

### v1.9 — 官方训练包接入（2026-11-15）
- 三个训练包 → zhixun/data/official/；官方 baselines 代码复制进 tools/baseline/
- load_platform_jsonl（复用官方 read_dataset 解析+展平+校验）
- gate.platform_metrics（与官方 train.metrics 逐位一致：分类 accuracy/macro_f1、回归 mae/rmse）
- 三包基线复算验证：A/B/C 指标与训练包记录完全一致


- 镜像交付完成：zhixun.tar（146.2 MB，linux/amd64）已导出；docker run 实测输出 0.8413/0.8442/0.9231
- Dockerfile：CMD 改 reproduce.py（默认入口=复现）；pip 源 ARG 化（默认官方，本地可传清华镜像）
- 核查报告：zhixun/核查报告.md（P0 全部修复/澄清 + 反伪造验证 + sha256）
- 测试：8/8 通过；SwanLab 云端项目 https://swanlab.cn/@GZLiang/smart-evolution
- 待外部资源：① 官方数据包（接 load_data）② PyTRiO/Qwen 额度（llm.enabled 改 true）

---

## 版本历史

### v1.8 — 镜像交付核查（2026-11-15）
- CMD main.py → reproduce.py（docker run 直接输出三指标；闭环仍可 python main.py）
- pip 源 ARG 化：默认官方 PyPI，--build-arg PIP_INDEX 可传镜像加速
- reproduce 诚实化：无历史记录时偏差输出 N/A（不伪造 0.0）
- .dockerignore 补 .git
- 反伪造验证：seed 42→0.8413/0.8442/0.9231；seed 43/100 结果变化；同种子完全一致；数据文件 sha256 已记录
- 镜像构建+运行+导出完成：zhixun.tar 146.2 MB，manifest 436d2868


- 容器化交付就绪：Dockerfile（python:3.10-slim + 中文locale + 时区 + UTF-8 强制）+ build_and_verify.bat + .dockerignore；镜像未 build（用户本机执行）
- 依赖 3.10 兼容性已按 PyPI 元数据核实（numpy 1.26.4 / scipy 1.13.1 / sklearn 1.5.2 / swanlab 0.10.5 / pyyaml 6.0.2；pytrio 不进镜像）
- 测试：8/8 通过（tests/run_all_tests.py）；回归任务冒烟通过（task.type=regression 全链路可跑）
- 运行：python main.py（full）、config 的 run.mode 改 dry-run（自检）、python reproduce.py（复现自检）
- SwanLab：已打通，云端项目 https://swanlab.cn/@GZLiang/smart-evolution（key 在 config 的 record.swanlab_*）
- 待外部资源：① 官方数据包（接 load_data）② PyTRiO/Qwen3.8-27B 额度（harness llm_decide 已就绪，config 的 llm.enabled 改 true 即可）③ 平台提交镜像
- 校赛交付：智训引擎-系统运行说明文档-校赛提交版.docx（工作区根目录）

---

## 版本历史

### v1.7 — 容器化交付（2026-11-14）
- Dockerfile：ENV DEBIAN_FRONTEND=noninteractive、中文 locale（zh_CN.UTF-8）、时区 Asia/Shanghai、PYTHONUTF8=1、清华镜像 pip、pytrio 不进镜像的说明注释
- requirements.txt：3.10 锁版本 + 兼容性注释（PyPI 元数据核实）
- build_and_verify.bat：5 步一键构建验证（build → python --version → pip list → reproduce.py → main.py），失败即停
- .dockerignore：排除 .venv/.idea/data/outputs，防开发环境进镜像


- 交接包：AI-HANDOFF.md（给其他 AI 的完整进度文档，接续工作先读它）
- 测试：8/8 通过（tests/run_all_tests.py）；回归任务冒烟通过（task.type=regression 全链路可跑）
- 运行：python main.py（full）、config 的 run.mode 改 dry-run（自检）、python reproduce.py（复现自检）
- SwanLab：已打通，云端项目 https://swanlab.cn/@GZLiang/smart-evolution（key 在 config 的 record.swanlab_*）
- 待外部资源：① 官方数据包（接 load_data）② PyTRiO/Qwen3.8-27B 额度（harness llm_decide 已就绪，config 的 llm.enabled 改 true 即可）③ 镜像构建与平台提交
- 校赛交付：智训引擎-系统运行说明文档-校赛提交版.docx（工作区根目录）

---

## 版本历史

### v1.6 — 回归任务支持 + 自检对齐官方三项（2026-11-14）
- 任务类型真正生效：compute_metrics 回归分支（rmse/mae/r2）；finetune fit/predict 支持回归（y 不做 ±1 映射，预测值即输出）
- gate 三层 + should_act 按主指标方向比较（rmse/mae 越小越好，其余越大越好）
- _self_check 对齐官方 3.1 三项：文件完整性 / 配置与字段合法性 / 数据契约过检
- SwanLab 同步改固定键（accuracy/f1/auc/rmse/mae/r2 + cost_gpu_hours/labels/tokens + cycle_id），消除稀疏曲线
- requirements 锁 swanlab==0.10.5


- 测试：8/8 通过（tests/run_all_tests.py）
- 运行：python main.py（full）、config 的 run.mode 改 dry-run（自检）、python reproduce.py（复现自检）
- SwanLab：已打通，云端项目 https://swanlab.cn/@GZLiang/smart-evolution（key 在 config 的 record.swanlab_*）
- 待外部资源：① 官方数据包（接 load_data）② PyTRiO/Qwen3.8-27B 额度（harness llm_decide 已就绪，config 的 llm.enabled 改 true 即可）③ 镜像构建与平台提交
- 校赛交付：智训引擎-系统运行说明文档-校赛提交版.docx（工作区根目录）

---

## 版本历史

### v1.5 — 审视收官（2026-11-14）
- 错误码 E1–E6：core/errors.py，8 个码挂载各降级路径
- requirements 环境标记：3.10 锁死 / 3.12+ 放开
- 测试补 strategy/harness/reproduce（5→8 项）
- gas 数据集多分类支持（load_gas_batch target_class=None）
- 模型层 GBDT（config model.algorithm，sklearn 不可用自动回退 linear）
- 新增本文件 VERSIONS.md

### v1.4 — 任务类型配置化
- 主指标不再硬编码 accuracy：config task.primary_metric / task.key_metrics，gate 三层 + should_act + 增益对照全部走配置
- README 重写对齐当前实现

### v1.3 — Harness Qwen3.8-27B 完整接口
- core/harness.py：提示词构造 / OpenAI 兼容调用 / JSON 解析 / 白名单校验
- config llm.base_url / llm.model；llm.enabled=true 时每轮问大模型，失败自动回退规则（已冒烟验证）

### v1.2 — 归因修正 + SwanLab 指标
- diagnose：概念漂移优先（pred_shift 先判），特征漂移改 PSI「或」KS（对齐官方表2/3.3）
- upload_swanlab：递归抽取嵌套数值（valid/accuracy、cost/used/gpu_hours 等上网页）

### v1.1 — SwanLab 云端打通
- config 填 API key + 项目 smart-evolution（SDK 只允许英文项目名）
- 实测上传成功；main.py 收尾自动 sync_run 同步全部记录；verify_swanlab.py 验证脚本

### v1.0 — 评测口径收官
- 7.1 预估 vs 实际增益对照（rollback 记录 gain{expected,actual,deviation,cost}）
- core/reproduce.py + reproduce.py（静态基线可复现，偏差 0.0）
- requirements 锁版本（py3.10）

### v0.9 — 记录对齐附录B + 超时看门狗
- 记录键名对齐官方附录B：run_id/trigger/data_snapshot/code/config/seed/cost/metrics/reason/artifacts/conclusion（+conclusion 三态）
- gate metrics 分 valid/replay/gate 三组
- 单阶段超时看门狗（StageTimeout + 每环节检查，config run.stage_timeout_seconds）

### v0.8 — 真实数据验证
- tools/data_loader：load_gas_batch（UCI gas 传感器 libsvm .dat）
- 实测 10 批次漂移检出：batch2 warning → batch3+ trigger
- 数据在 data/gas/，临时脚本已清

### v0.7 — 策略层 + Harness 骨架 + 收尾加固
- core/strategy.py（策略记忆：历史动作成功率）
- core/harness.py 初版（白名单 + llm_decide 占位）
- validate_segment 数据契约校验 + 批次异常跳过
- logger flush_pending/upload_swanlab（SwanLab 补传）
- 每轮检查点（rollback.persist）+ Dockerfile

### v0.6 — dry-run + 退出码 + 三步收尾
- run.mode: dry-run、_self_check、_finalize（存最优/摘要/决策日志）
- 退出码：0 正常 / 2 预算触顶 / 3 自检失败

### v0.5 — 漂移监测六类 + 连续批次
- monitor 六类对象（PSI/KS/类别分布/缺失越界重复/预测分布/延迟标签/服务健康）
- drift.consecutive=2 连续批次确认

### v0.4 — 门禁第二层 + 归因卡7字段
- gate 第二层：回放段 vs 静态基线（gate.baseline=registry）
- diagnose 7 字段卡（类型/子群/证据/置信度/动作/预估收益/预估成本）

### v0.3 — 预算修复 + 配置对齐官方
- budget：max_gpu_hours/max_labels/max_tokens（官方命名）+ 每动作扣费
- config.yaml 对齐官方附录A 模板（run/drift/budget/rsi/rerun/gate/record/llm）

### v0.2 — 闸门 + 门禁三层 + 归因
- should_act（只判断不动手）；gate 三层（绝对指标/抗漂移/成本）；多指标一致性 warning
- diagnose 规则归因（data_drift/concept_drift/no_drift）

### v0.1 — 初始骨架
- 五环节闭环（监测→诊断→优化→门禁→回滚）+ 假数据自洽生成
- 实验记录 10 字段 + 本地 JSONL + 待补传队列
- 故障注入降级（训练/诊断/门禁/记录）

---

## 工作约定（节约上下文）

- 每轮修改：改完在本文件追加「### vX.Y — 标题（日期）」条目
- 新会话接续：先读本文件 + README + config.yaml 即可恢复全部上下文
- 代码细节以文件为准，对话里不再重复叙述
