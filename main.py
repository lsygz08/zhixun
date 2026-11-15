"""智训引擎入口：一键启动整个"漂移监测→诊断归因→实验优化→评估上线→回滚"闭环。

运行模式（run.mode）：full 正式运行；dry-run 空转自检。
退出码：0 正常收尾；2 预算触顶的受控收尾；3 自检失败中止。
直接运行：python main.py
"""
import os
import sys
import time
import traceback

# 保证无论从哪个目录启动，都能 import 到 core / tools / record / budget
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import cfg_get, load_config
from core import diagnose, gate, harness, monitor, optimize, rollback, strategy
from tools import data_loader, finetune
from record import logger
from budget.tracker import BudgetTracker


def _print_config_summary(config):
    """启动时打印配置摘要。"""
    print("=" * 64)
    print("智训引擎 配置摘要")
    print("=" * 64)
    for section in ("run", "drift", "budget", "rsi", "rerun", "gate", "record", "llm",
                    "cost", "optimize", "data", "rollback", "faults"):
        print(f"  [{section}]")
        for k, v in (config.get(section) or {}).items():
            print(f"    {k}: {v}")
    print("=" * 64)


def predict_on_segment(segment, model):
    """用当前模型给数据段打预测，把 pred / scores 填回数据段。返回数据段。"""
    pred, scores = finetune.predict(model, segment["X"])
    segment["pred"] = pred
    segment["scores"] = scores
    return segment


class StageTimeout(Exception):
    """单阶段超时（看门狗）。"""


def _check_deadline(deadline, iteration, stage):
    """# new：超时看门狗检查，超时抛 StageTimeout。"""
    if deadline is not None and time.time() > deadline:
        raise StageTimeout(f"第 {iteration} 轮「{stage}」阶段超时")


def run_cycle(segment, current_model, config, budget, iteration, memory=None, deadline=None):
    """严格串联五个环节：监测 → 诊断 → 优化 → 门禁 → 回滚。

    返回更新后的当前最优模型。
    """
    seed = cfg_get(config, "run.seed", 42) + iteration

    # ---- 环节 1：漂移监测 ----
    _check_deadline(deadline, iteration, "监测")
    stats = monitor.drift_stats(segment, config=config)
    logger.log("monitor", detail={"stats": stats},
               data_hash=logger.hash_segment(segment), resource=budget.summary())
    print(f"[cycle {iteration}][monitor] level={stats['level']} "
          f"psi={stats['psi']} ks={stats['ks']} missing={stats['missing_rate']} "
          f"pred_shift={stats['pred_shift']} consecutive={stats.get('consecutive_count', 0)}")

    # ---- 环节 2：诊断归因（故障降级·2：异常则切纯规则策略）----
    _check_deadline(deadline, iteration, "诊断")
    card = None
    fallback_plan = None
    try:
        card = diagnose.attribute(stats, config)
    except diagnose.DiagnoseError as e:
        print(f"[cycle {iteration}][diagnose] 归因模块异常，切纯规则策略：{e}")
        fallback_plan = optimize.rule_fallback_plan(stats, config)
        print(f"[cycle {iteration}][diagnose] 纯规则兜底 plan -> action={fallback_plan['action']}，{fallback_plan['reason']}")
        logger.log("diagnose", detail={"error": str(e), "fallback_plan": fallback_plan, "error_code": "E5001"},
                   decision_reason="归因模块异常，降级纯规则策略。", resource=budget.summary())
    else:
        logger.log("diagnose", detail={"card": card},
                   trigger_reason=card.get("drift_type"), decision_reason=card.get("reason"),
                   resource=budget.summary())
        print(f"[cycle {iteration}][diagnose] {card['drift_type']} / {card['severity']} / "
              f"conf={card['confidence']} -> {card['suggested_action']}")

    # ---- 闸门：该不该动（should_act，只判断、不动手、不改状态）----
    _check_deadline(deadline, iteration, "该不该动")
    if card is not None:
        diag = card
    else:
        diag = {"level": stats.get("level", "normal"),
                "suggested_action": (fallback_plan or {}).get("action", "no_action")}
    should, should_reason = gate.should_act(diag, current_model, segment, config)
    logger.log("should_act", detail={"diagnosis": diag, "should_act": should},
               decision_reason=should_reason, resource=budget.summary())
    if not should:
        print(f"[cycle {iteration}][should_act] 不动：{should_reason}")
        return current_model
    print(f"[cycle {iteration}][should_act] 动作：{should_reason}")

    # ---- 环节 3：优化（动作 + 超参建议 + 成本收益）----
    _check_deadline(deadline, iteration, "优化")
    plan = fallback_plan if fallback_plan is not None else optimize.suggest(card, config, memory)

    # Harness 大模型决策（llm.enabled 时尝试；不可用/校验失败自动回退规则）
    if card is not None and plan.get("action") != "no_action":
        llm_plan = harness.decide_plan(card, stats, config, budget)
        if llm_plan is not None:
            plan = llm_plan

    if plan["action"] == "no_action":
        logger.log("optimize", detail={"plan": plan}, decision_reason=plan["reason"],
                   resource=budget.summary())
        print(f"[cycle {iteration}][optimize] 无漂移，本轮不动作。")
        return current_model

    ok, cb_reason = optimize.cost_benefit_check(plan, budget, config)
    if not ok:
        logger.log("optimize", detail={"plan": plan, "cost_benefit": cb_reason},
                   decision_reason=cb_reason, resource=budget.summary())
        print(f"[cycle {iteration}][optimize] 成本收益不划算，跳过：{cb_reason}")
        return current_model

    # 执行动作：按 plan['action'] 分发（concept_drift -> finetune；data_drift -> 重采样+校准）
    candidate = None
    try:
        candidate = optimize.execute(plan, current_model, segment, seed=seed, config=config)
    except finetune.FinetuneError as e:
        # 故障降级·1：微调失败 -> 降档重试 -> 仍失败放弃候选
        print(f"[cycle {iteration}][optimize] 微调失败，降档重试：{e}")
        try:
            candidate = optimize.execute(plan, current_model, segment, seed=seed,
                                         downgrade=True, config=config)
        except finetune.FinetuneError as e2:
            print(f"[cycle {iteration}][optimize] 降档后仍失败，放弃候选：{e2}")
            candidate = None
    if candidate is None:
        logger.log("optimize", detail={"action": plan["action"], "status": "candidate_abandoned", "error_code": "E2002"},
                   decision_reason="动作执行失败且降档重试仍失败，放弃本轮候选。", resource=budget.summary())
        return current_model

    # 记账：消耗预算（resample 只扣 resample_labels，finetune 扣 gpu/tokens，按 [cost] 累加）
    action_cost_val = optimize.action_cost(plan["action"], config)
    budget.spend(action_cost_val)
    print(f"[cycle {iteration}][budget] 扣费 {action_cost_val}，剩余额度 {budget.summary()['remaining']}")

    # 计算候选在漂移段上的指标并落产物
    cand_pred, cand_scores = finetune.predict(candidate, segment["X"])
    candidate["metric"] = gate.compute_metrics(segment["y"], cand_pred, cand_scores,
                                               cfg_get(config, "task.type", "classification"))
    artifact_path = finetune.save_artifact(candidate, logger.artifacts_dir())
    candidate["artifact"] = artifact_path
    logger.log("optimize", detail={"plan": plan, "candidate": candidate["version"],
                                   "candidate_metric": candidate["metric"]},
               eval_result=candidate["metric"], decision_reason=plan["reason"],
               artifact_path=artifact_path, resource=budget.summary())

    # ---- 环节 4：评估门禁（故障降级·3：门禁异常回滚；·4：评测不可用本地回放集）----
    _check_deadline(deadline, iteration, "门禁")
    try:
        gate_result = gate.evaluate(candidate, current_model, budget, segment, config)
    except gate.GateError as e:
        # 降级·3：门禁模块异常 -> 默认不通过、回滚
        print(f"[cycle {iteration}][gate] 门禁异常，降级为不通过：{e}")
        gate_result = {"pass": False, "checks": {}, "warnings": [],
                       "reason": "门禁异常已降级：默认不通过，走回滚。", "degraded": "gate_error",
                       "error_code": "E3001"}
    except Exception as e:
        # 降级·4：评测接口不可用 -> 本地漂移回放集 -> 待补测
        print(f"[cycle {iteration}][gate] 评测接口不可用，改用本地漂移回放集：{e}")
        gate_result = gate.fallback_evaluate(candidate, segment, config)
        gate_result["error_code"] = "E3002"
    gate_metrics = {
        "valid": candidate.get("metric") or {},
        "replay": gate_result.get("replay_metric") or {},
        "gate": {"result": "pass" if gate_result["pass"] else "fail",
                 "layers": gate_result.get("checks", {})},
    }
    logger.log("gate", detail={"gate": gate_result}, metrics=gate_metrics,
               decision_reason=gate_result["reason"], resource=budget.summary(),
               conclusion="采纳" if gate_result["pass"] else "放弃")
    print(f"[cycle {iteration}][gate] pass={gate_result['pass']} -> {gate_result['reason']}")

    # ---- 7.1 客观分口径：预估 vs 实际增益对照（偏差入记录，供策略记忆）----
    primary_metric = cfg_get(config, "task.primary_metric", "accuracy")
    base_acc = float((current_model.get("metric") or {}).get(primary_metric, 0.0))
    cand_acc = float((candidate.get("metric") or {}).get(primary_metric, 0.0))
    expected_gain = float(plan.get("expected_gain", 0.0))
    actual_gain = cand_acc - base_acc
    gain_deviation = actual_gain - expected_gain

    # ---- 环节 5：上线 / 回滚 ----
    if gate_result["pass"]:
        new_model = rollback.promote(candidate)
        decision = f"上线 {candidate['version']}"
    else:
        new_model = rollback.rollback()
        decision = f"回滚，保持 {new_model['version']}"
    logger.log("rollback",
               detail={"promoted": candidate["version"] if gate_result["pass"] else None,
                       "current_best": new_model["version"],
                       "gain": {"expected": round(expected_gain, 4),
                                "actual": round(actual_gain, 4),
                                "deviation": round(gain_deviation, 4),
                                "cost": optimize.action_cost(plan["action"], config)}},
               decision_reason=decision, artifact_path=artifact_path,
               resource=budget.summary(),
               conclusion="采纳" if gate_result["pass"] else "回滚")
    print(f"[cycle {iteration}][rollback] {decision}（候选 {candidate['version']} "
          f"指标 {candidate['metric']}，最优 {new_model['version']}）")
    return new_model


def _self_check(config):
    """# modified：启动自检——对齐官方 3.1 三项：文件完整性 / 字段与契约 / 配置合法性。"""
    problems = []
    here = os.path.dirname(os.path.abspath(__file__))
    # ① 文件完整性
    for f in ("main.py", "config.yaml", "requirements.txt"):
        if not os.path.exists(os.path.join(here, f)):
            problems.append(f"关键文件缺失 [{f}]")
    # ② 依赖
    try:
        import numpy  # noqa: F401
    except Exception as e:
        problems.append(f"numpy 不可用: {e}")
    # ③ 配置段与字段合法性
    for key in ("drift", "budget", "gate", "rsi", "task"):
        if not isinstance(config.get(key), dict):
            problems.append(f"配置缺少段 [{key}]")
    if cfg_get(config, "budget.max_labels", 0) <= 0:
        problems.append("budget.max_labels 必须 > 0")
    tt = cfg_get(config, "task.type", "classification")
    pm = cfg_get(config, "task.primary_metric", "accuracy")
    if tt not in ("classification", "regression"):
        problems.append(f"task.type 非法：{tt}")
    if tt == "classification" and pm not in ("accuracy", "f1", "auc"):
        problems.append(f"分类任务主指标非法：{pm}")
    if tt == "regression" and pm not in ("rmse", "mae", "r2"):
        problems.append(f"回归任务主指标非法：{pm}")
    # ④ 数据契约（造一段假数据过契约校验，模拟"字段与契约一致性"）
    seg_ok, seg_reason = data_loader.validate_segment(
        data_loader.make_fake_segment(n_samples=64, n_features=8,
                                      seed=cfg_get(config, "run.seed", 42)), config)
    if not seg_ok:
        problems.append(f"数据契约校验失败：{seg_reason}")
    return (len(problems) == 0, "；".join(problems))


def _finalize(budget, run_id):
    """# new：三步收尾——①保存当前最优版本 ②写门禁报告占位 ③输出决策日志/摘要。"""
    best_path = rollback.persist(logger.artifacts_dir())
    best_model = rollback.best()
    print("=" * 64)
    print("运行摘要（三步收尾）")
    print(f"  最优版本：{best_model['version'] if best_model else None}")
    print(f"  最优指标：{best_model['metric'] if best_model else None}")
    print(f"  预算消耗：{budget.summary()}")
    print(f"  产物路径：{best_path}")
    print(f"  实验记录：{logger.experiments_path(run_id)}")
    print("=" * 64)
    logger.log("finalize", detail={"best_version": best_model["version"] if best_model else None,
                                   "budget": budget.summary()},
               decision_reason="三步收尾完成：已保存最优版本、输出摘要与决策日志。")
    # 同步到 SwanLab（best-effort，离线时静默跳过）
    try:
        logger.sync_run(run_id)
    except Exception:
        pass
    return best_path


def main():
    """主流程：自检 -> （dry-run）空转 -> 正式运行 -> 三步收尾。返回退出码。"""
    config = load_config()
    _print_config_summary(config)

    run_id = logger.init(run_id=cfg_get(config, "run.run_id", "") or None,
                         config=config, seed=cfg_get(config, "run.seed", 42))
    print(f"[main] 本次运行 ID：{run_id}")

    # 第一步：自检
    ok, reason = _self_check(config)
    if not ok:
        print(f"[main] 自检失败：{reason}")
        logger.log("self_check", detail={"error": reason}, decision_reason="自检失败，中止（退出码 3）。")
        return 3

    mode = cfg_get(config, "run.mode", "full")
    budget = BudgetTracker(cfg_get(config, "budget", {}))
    monitor.reset_baseline()
    rollback.reset()
    memory = strategy.load_memory(logger.experiments_dir())   # 策略记忆（历史实验）

    # 基线：用第一段正常数据构造初始模型
    baseline_segment = data_loader.make_fake_segment(
        n_samples=cfg_get(config, "data.n_samples", 800),
        n_features=cfg_get(config, "data.n_features", 8),
        seed=cfg_get(config, "run.seed", 42),
        drift=False, name="baseline")
    current_model = finetune.make_initial_model(baseline_segment)
    predict_on_segment(baseline_segment, current_model)
    current_model["metric"] = gate.compute_metrics(
        baseline_segment["y"], baseline_segment["pred"], baseline_segment["scores"],
        cfg_get(config, "task.type", "classification"))
    current_model["baseline_metrics"] = dict(current_model["metric"])   # 固定基线参照，供 should_act 对比
    current_model["artifact"] = finetune.save_artifact(current_model, logger.artifacts_dir())
    monitor.set_baseline(baseline_segment)
    rollback.register_best(current_model)
    logger.set_cycle(0)   # 基线记作第 0 轮
    logger.log("baseline",
               detail={"model_version": current_model["version"], "metric": current_model["metric"]},
               eval_result=current_model["metric"], artifact_path=current_model["artifact"],
               resource=budget.summary(), conclusion="基线")
    print(f"[main] 基线模型 {current_model['version']} 就绪，指标 {current_model['metric']}")

    # 第二步：dry-run 模式，跑一轮空转闭环以验证链路
    if mode == "dry-run":
        print("[main] dry-run 模式：执行一轮空转闭环以验证链路")
        dry_seg = data_loader.make_fake_segment(
            n_samples=cfg_get(config, "data.n_samples", 800),
            n_features=cfg_get(config, "data.n_features", 8),
            seed=cfg_get(config, "run.seed", 42) + 1,
            drift=True, name="dry_run")
        predict_on_segment(dry_seg, current_model)
        run_cycle(dry_seg, current_model, config, budget, 0, memory)
        _finalize(budget, run_id)
        return 0

    # 第三步：正式运行
    stopped_by_budget = False
    max_iter = int(cfg_get(config, "rsi.max_rounds", 8))
    for it in range(1, max_iter + 1):
        logger.set_cycle(it)   # 记录当前轮次号
        # 预算检查 + 迭代上限检查：触顶即停
        if not budget.check({}):
            stopped_by_budget = True
            print("[main] 预算触顶，停止迭代。")
            break

        segment = data_loader.make_fake_segment(
            n_samples=cfg_get(config, "data.n_samples", 800),
            n_features=cfg_get(config, "data.n_features", 8),
            seed=cfg_get(config, "run.seed", 42) + it,
            drift=cfg_get(config, "data.drift", True),
            drift_severity=cfg_get(config, "data.drift_severity", "medium"),
            drift_affects_performance=cfg_get(config, "data.drift_affects_performance", True),
            name=f"segment_{it}")

        # 数据契约校验：异常批次跳过（E1xxx 数据批次异常）
        seg_ok, seg_reason = data_loader.validate_segment(segment, config)
        if not seg_ok:
            print(f"[main] 第 {it} 轮数据批次异常，跳过：{seg_reason}")
            logger.log("data_batch_skip", detail={"iteration": it, "error": seg_reason, "error_code": "E1002"},
                       decision_reason="数据契约校验失败，跳过该批次。", resource=budget.summary())
            continue

        predict_on_segment(segment, current_model)

        # 单阶段超时看门狗
        deadline = time.time() + float(cfg_get(config, "run.stage_timeout_seconds", 60))
        try:
            current_model = run_cycle(segment, current_model, config, budget, it, memory, deadline)
            rollback.persist(logger.artifacts_dir())   # 检查点：每轮结束保存当前最优版本
        except StageTimeout as e:
            # 表6：单阶段超时 -> 保存检查点并早停
            print(f"[main] 单阶段超时，保存检查点并早停：{e}")
            rollback.persist(logger.artifacts_dir())
            logger.log("stage_timeout", detail={"iteration": it, "error": str(e), "error_code": "E6001"},
                       decision_reason="单阶段超时：保存检查点并早停。", resource=budget.summary())
        except Exception as e:
            # 最外层单轮兜底：任何未捕获异常进入降级路径，继续下一轮
            print(f"[main] 第 {it} 轮异常，进入降级路径并继续：{e}")
            traceback.print_exc()
            logger.log("cycle_error", detail={"iteration": it, "error": str(e), "error_code": "E6002"},
                       decision_reason="本轮异常，降级继续。", resource=budget.summary())

    # 第四步：三步收尾
    _finalize(budget, run_id)

    # 退出码：预算触顶 -> 2；正常完成 -> 0
    if stopped_by_budget:
        print("[main] 受控收尾：预算触顶（退出码 2）。")
        return 2
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as e:
        # 最外层兜底：绝不静默崩溃
        print(f"[main] 严重异常，进入兜底降级路径：{e}")
        traceback.print_exc()
        try:
            logger.log("fatal", detail={"error": str(e)}, decision_reason="最外层兜底降级。")
        except Exception:
            pass
        code = 1
    print(f"[main] 退出码：{code}")
    sys.exit(code)
