# 智训引擎（zhixun）

「工业模型自进化系统」：数据持续漂移条件下，无人干预地自动维持并提升模型性能，受控递归自改进（RSI）。

## 闭环

monitor（漂移监测）→ diagnose（归因卡7字段）→ should_act（该不该动）→ optimize（动作分发）→ gate（三层门禁）→ rollback（上线/回滚），全程留痕。

## 快速开始

    pip install -r requirements.txt
    python main.py            # full 模式
    python main.py --dry-run  # 不存在的参数；dry-run 用 config run.mode: dry-run
    python reproduce.py       # 静态基线可复现自检
    python tests/run_all_tests.py   # 测试套件

## 结构

    core/        monitor/diagnose/harness/optimize/gate/rollback/strategy/reproduce
    tools/       data_loader（假数据 + gas 传感器 .dat）/ finetune（线性模型 + 微调接口）
    record/      logger（记录对齐官方附录B + SwanLab 同步 + 本地队列补传）
    budget/      tracker（gpu/labels/tokens 三项预算）
    tests/       5 个测试 + run_all_tests 入口
    data/gas/    UCI 气体传感器漂移数据集（练习用）
    outputs/     experiments（JSONL 记录）/ artifacts（模型产物）

## 对齐官方文档的要点

- 配置：附录A 模板（run/drift/budget/rsi/rerun/gate/record/llm + 扩展 task/cost/data/faults）
- 漂移监测：表2 六类（PSI/KS检验/类别分布/缺失越界重复/预测分布/延迟标签/服务健康）+ 连续批次
- 归因卡：7 字段（类型/子群/证据/置信度/动作/预估收益/预估成本）
- 门禁：三层（绝对指标 / 抗漂移vs静态基线 / 成本）+ 多指标一致性 warning
- 记录：附录B 字段（run_id/trigger/data_snapshot/code/config/seed/cost/metrics/reason/artifacts/conclusion）
- 故障分级：表6 全部六类降级路径
- 运行：自检→dry-run→正式→三步收尾；退出码 0/2/3
- Harness：Qwen3.8-27B 接口（llm.enabled=true 启用，失败回退规则）
- 策略层：策略记忆（读历史实验学动作成功率）

## 待接入（等外部资源）

- 官方数据包（load_data 接真实格式 + 契约校验）
- PyTRiO/Qwen 额度（harness 大模型调用）
- SwanLab SDK 已接（config record.swanlab_api_key / swanlab_project）
