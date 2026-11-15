# 智训引擎 AI 交接包（AI Handoff）

写给任何接手的 AI/成员的自包含进度文档。先读本文件，再看 VERSIONS.md 和 README.md。

## 0. 一分钟速览

- 赛题：第二届北京市大学生"人工智能+"创新大赛 ——「智训引擎：工业制造场景下的模型训练与超参数优化」
- 目标：数据持续漂移条件下，无人干预、离线、预算受限、全程留痕地自动维持并提升模型性能（受控递归自改进 RSI）
- 进度：代码侧全部完成（VERSIONS v1.6），测试 8/8 通过，校赛文档已交付（2026-11-15 截止），正等官方数据包与 PyTRiO 训推额度
- 一句话原理：五环节闭环（漂移监测→诊断归因→should_act→实验优化→门禁→上线/回滚），两层自进化（模型层自迭代 + 策略层自改进），大模型只判断、动作由确定性工具执行

## 1. 位置与环境

- 项目根：C:\Users\STEVE\Documents\deepseek-harness\default-workspace\zhixun\
- 用户 Python：C:\Users\STEVE\AppData\Local\Python\pythoncore-3.14-64\python.exe（3.14.7；系统 python 命令被 Microsoft Store 别名劫持，PowerShell 里用 $py 变量指向完整路径）
- 校赛文档（已填实测值+赛段）：智训引擎-系统运行说明文档-校赛提交版.docx（工作区根目录）
- 官方模板：智训引擎-系统运行说明文档.docx（V1.0，评审基准，代码已对齐）
- pip 默认清华镜像；Scripts 目录不在 PATH（trio.exe 等要完整路径调用）

## 2. 完成度

已完成：
- 五环节闭环 + should_act 闸门（只判断不动手，严格串行）
- 漂移监测六类（PSI/KS检验/类别分布/缺失越界重复/预测分布/延迟标签/服务健康）+ 连续批次确认（consecutive=2）
- 归因卡 7 字段（类型/子群/证据/置信度/动作/预估收益/预估成本）；概念漂移优先，特征漂移用 PSI「或」KS
- 门禁三层（绝对指标+min_gain / 抗漂移vs静态基线 / 成本合规）+ 多指标一致性 warning
- 实验记录对齐官方附录B（run_id/trigger/data_snapshot/code/config/seed/cost/metrics/reason/artifacts/conclusion）+ SwanLab 同步 + 本地队列补传
- 预算三项（max_gpu_hours 48 / max_labels 2000 / max_tokens 2000000）+ 每动作成本收益判断
- 故障分级表6 全部：训练降档重试/单阶段超时看门狗/评测回放集待补测/记录本地队列/推理后端切规则/数据批次跳过
- 错误码 E1-E6（core/errors.py，8 个码已挂载各降级路径）
- 策略层自改进（core/strategy.py：从历史 JSONL 学动作成功率）
- Harness 大模型接口（core/harness.py：Qwen3.8-27B，OpenAI 兼容调用+白名单校验+规则回退）
- dry-run + 退出码（0 正常/2 预算触顶/3 自检失败）+ 三步收尾 + 每轮检查点
- 任务类型分类/回归双支持（主指标可配置，rmse/mae 越低越好方向已处理）
- 模型层 linear/GBDT 双算法（sklearn 不可用自动回退）
- 真实数据验证：UCI gas 传感器 10 批次漂移检出（data/gas/）
- 复现脚本 reproduce.py（同种子偏差 0.0）
- 测试 8 项全部通过（tests/run_all_tests.py）

待外部资源（代码接口已全部预留）：
- 官方数据包（11 月初开放，等平台发放）→ 接 tools/data_loader.load_data
- PyTRiO 训推额度 / Qwen3.8-27B 权重 → config 的 llm.enabled 改 true 即启用大模型决策
- 镜像构建与平台提交（Dockerfile 已就绪，python:3.10-slim）

## 3. 目录结构

    main.py            入口：自检→dry-run→8轮闭环→三步收尾→SwanLab同步
    config.yaml        官方附录A模板 + 扩展段（task/model/cost/data/faults）
    VERSIONS.md        版本历史（v0.1→v1.6），每次修改追加条目
    README.md          快速上手
    AI-HANDOFF.md      本文件
    Dockerfile / requirements.txt（版本已按 3.10/3.12+ 分段锁定）
    reproduce.py       静态基线复现自检
    verify_swanlab.py  SwanLab 连接验证
    core/              monitor/diagnose/harness/optimize/gate/rollback/strategy/reproduce/errors
    tools/             data_loader（假数据+gas .dat）/finetune（线性+GBDT+微调接口）
    record/            logger（附录B字段+SwanLab+队列补传）
    budget/            tracker（三项预算+扣费）
    tests/             8 个测试 + run_all_tests 入口
    data/gas/          UCI 气体传感器漂移数据集（10 批次）
    outputs/           experiments（JSONL 记录）/artifacts（模型产物）

## 4. 关键设计（对齐官方文档）

- 3.1 数据契约：validate_segment（字段/类型/缺失率/标签）+ 启动自检三项（文件完整性/配置字段/契约过检）
- 3.3 归因：pred_shift 优先判概念漂移；特征漂移 PSI「或」KS（对齐表2）
- 3.6 门禁：三层 + f1/auc 回退降级 warning 不拒判
- 4.x Harness：大模型只输出白名单动作（resample_or_calibrate/finetune/no_action），执行器校验 schema；先规则后智能
- 5.x 记录：字段对齐附录B；SwanLab 固定键上传（accuracy/f1/auc/rmse/mae/r2 + cost_gpu_hours/labels/tokens + cycle_id）
- 7.x 评测：7.1 预估vs实际增益对照入记录；7.2 三门槛；7.3 九项自检（代码项已齐，镜像/版本绑定待构建）
- 10.1 错误码：E1 数据 / E2 训练 / E3 门禁评测 / E4 记录 / E5 推理 / E6 其他

## 5. 关键配置值（config.yaml）

- drift：psi_warn 0.1 / psi_trigger 0.2 / ks_alpha 0.05 / consecutive 2 / missing 0.05、0.15 / pred_shift 0.1、0.2
- budget：48 / 2000 / 2000000；cost：finetune_gpu 0.2、finetune_tokens 1000、resample_labels 200、llm_tokens 500
- rsi：max_rounds 8、max_experiments_per_round 4、max_retrain_per_model 2；rerun.max_attempts 2
- gate：baseline registry、min_gain 0.0、severe_regression_margin 0.15、accuracy_floor_ratio 0.95
- task：classification / primary_metric accuracy / key_metrics [f1, auc]
- model：linear（可选 gbdt）
- llm：backend local、model_path /opt/models/qwen3.8-27b、fallback_policy rule、enabled false、base_url http://localhost:8000/v1
- record：backend swanlab、fallback local_queue、swanlab_api_key（已填）、swanlab_project smart-evolution
- data：n_samples 800、n_features 8、drift true、medium、affects_performance true
- faults：inject_finetune_failure / inject_diagnose_failure / inject_gate_failure（演示故障注入开关）

## 6. 运行方式

    cd C:\Users\STEVE\Documents\deepseek-harness\default-workspace\zhixun
    $py = "C:\Users\STEVE\AppData\Local\Python\pythoncore-3.14-64\python.exe"
    & $py main.py                     # 正式运行（收尾自动同步 SwanLab）
    & $py tests/run_all_tests.py      # 8 项测试
    & $py reproduce.py                # 复现自检
    & $py verify_swanlab.py           # SwanLab 连接验证

dry-run：config.yaml 里 run.mode 改 dry-run。

## 7. 账号与链接

- SwanLab：云端项目 https://swanlab.cn/@GZLiang/smart-evolution（账号 Liang，API key 在 config record.swanlab_api_key）
- PyTRiO：pytrio 0.2.9 已安装，已登录（账号 Phoenix_Dove）；API key 在 https://pytrio.com 个人主页；CLI 是 Scripts 目录下 trio.exe（完整路径 C:\Users\STEVE\AppData\Local\Python\pythoncore-3.14-64\Scripts\trio.exe）
- 数据源（练习）：UCI gas 传感器漂移 https://archive.ics.uci.edu/dataset/224/gas ；XJTU-SY 轴承（故障诊断）；CMAPSS（RUL 回归）

## 8. 下一步（按优先级）

1. 校赛文档封面补 4 项（队伍/学校/成员/指导教师）后提交（2026-11-15 截止）
2. 官方数据包发放后：接 load_data + 契约校验 + dry-run 验证
3. PyTRiO 额度下发后：起 Qwen3.8-27B 本地推理服务，config llm.enabled 改 true、base_url 填服务地址
4. 本地 docker build 验证镜像离线启动，按平台规范提交
5. 平台开放后跑 full 自检九项，回填文档表10/11 实测值

## 9. 已知约定与坑

- python 命令被 Store 别名劫持，必须用完整路径（$py）
- pip 安装的 Scripts（trio.exe 等）不在 PATH
- agent 沙箱验证用 bundled Python 3.12（无 scipy/sklearn），代码全部有 numpy 兜底
- SwanLab SDK 的 project 名只允许英文/数字（智造进化论 → smart-evolution）
- 对话上下文可能被压缩，以本文件 + VERSIONS.md + 代码文件为准
- 每次代码修改后：跑 tests/run_all_tests.py + 在 VERSIONS.md 追加版本条目
