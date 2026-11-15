"""平台入口：从官方检查点开始，N 轮自主续训闭环。

每轮：读历史 → 诊断（漂移+指标轨迹）→ 决策（epochs/学习率）→ 实验（续训）
      → 门禁（验证集主指标 vs 当前最优，方向感知）→ 采纳/回滚 → 记录。
对应官方要求：每个数据集 10 轮优化；至少 3 次"读历史—做实验—依据结果决策"记录；
效果不低于平台基线；无人干预。

用法：
    python main_platform.py --pack data/official/cpu-a-guide-20261110-v1 \
        --profile A --rounds 10 --output outputs/platform/A
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tools.baseline import engine
from tools.baseline.model import Model
from core import load_config
from core import harness
from core import monitor
from record import logger


def is_lower_better(metric):
    return metric in ("rmse", "mae")


def gain(cand, base, metric):
    return (base - cand) if is_lower_better(metric) else (cand - base)


def fill_predictions(model, seg):
    """把模型预测填入数据段（monitor 用）。分类：pred=argmax，scores=最大概率；回归：pred=scores=预测值。"""
    probas = model.predict(seg["X"])
    if model.meta["task"] == "classification":
        seg["pred"] = probas.argmax(axis=1).astype(np.int64)
        seg["scores"] = probas.max(axis=1)
    else:
        seg["pred"] = probas
        seg["scores"] = probas
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    config = load_config()
    run_id = logger.init(run_id=args.run_id or None, config=config,
                         seed=int((config.get("run") or {}).get("seed", 42)))
    print(f"[platform] run={run_id} profile={args.profile} rounds={args.rounds}")

    model, cfg, train, val = engine.load_profile(args.pack, args.profile)
    task = cfg["task"]
    primary = "macro_f1" if task == "classification" else "rmse"

    os.makedirs(args.output, exist_ok=True)
    best_path = os.path.join(args.output, "best_checkpoint.npz")

    def save_best(m):
        if os.path.exists(best_path):
            os.remove(best_path)   # 官方 save 用独占创建，覆盖前先删
        m.save(best_path)

    # ---- 基线 ----
    base_metrics = engine.platform_metrics(model, val)
    save_best(model)
    logger.set_cycle(0)
    logger.log("baseline", detail={"profile": args.profile, "metrics": base_metrics},
               eval_result=base_metrics, conclusion="基线", artifact_path=best_path)
    print(f"[platform] 基线 {primary}={base_metrics.get(primary):.6f}")

    # ---- 漂移监测（训练 vs 验证 特征分布，作为每轮证据）----
    train_seg = fill_predictions(model, {"name": "train", "X": train["x"], "y": train["y"],
                                         "features": train["paths"],
                                         "pred": None, "scores": None,
                                         "meta": {"n_samples": len(train["y"])}})
    val_seg = fill_predictions(model, {"name": "validation", "X": val["x"], "y": val["y"],
                                       "features": val["paths"],
                                       "pred": None, "scores": None,
                                       "meta": {"n_samples": len(val["y"])}})
    monitor.set_baseline(train_seg)
    drift = monitor.drift_stats(val_seg, config=config)
    logger.log("monitor", detail={"stats": drift, "note": "训练vs验证特征分布"},
               data_hash=val["sha256"])
    print(f"[platform] 漂移监测 psi={drift['psi']:.4f} ks={drift['ks']:.4f} level={drift['level']}")

    best_metrics = dict(base_metrics)
    history = []   # (轮次, 主指标, 决策, 结论)
    lr = float(cfg.get("learning_rate", 0.003))
    epochs = int(args.epochs)
    degrade_streak = 0
    max_seconds = float((config.get("platform") or {}).get("max_training_seconds", 1800))
    total_seconds = 0.0

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        logger.set_cycle(r)

        # 预算：总训练时长触顶即受控收尾
        if total_seconds >= max_seconds:
            logger.log("budget_stop", detail={"total_seconds": round(total_seconds, 1),
                                              "limit": max_seconds},
                       decision_reason=f"训练时长触顶（{total_seconds:.0f}s/{max_seconds:.0f}s），受控收尾。",
                       conclusion="停训")
            print(f"[platform] 训练时长触顶，第 {r} 轮起停止。")
            break

        # ---- 环节 1：读历史 + 诊断 + 决策 ----
        recent = history[-3:]
        decision = {"epochs": epochs, "learning_rate": lr}
        reason = f"第 {r} 轮："
        if not recent:
            reason += "首轮探索，按默认参数续训。"
        elif degrade_streak >= 2:
            lr = max(lr * 0.5, 0.0001)
            epochs = max(epochs - 1, 1)
            decision.update(epochs=epochs, learning_rate=lr)
            reason += f"连续 {degrade_streak} 轮劣化（历史：{[(h[0], round(h[1], 4)) for h in recent]}），"
            reason += f"降低学习率至 {lr} 并减少轮数，模拟过拟合恢复。"
            degrade_streak = 0
        else:
            last_gain = history[-1][3] if history else 0.0
            if last_gain >= 0 and last_gain < 0.0005 * max(abs(history[-1][1]), 1e-9):
                lr = max(lr * 0.5, 0.0001)
                decision.update(learning_rate=lr)
                reason += f"上轮增益微弱（{history[-1][1]:.6f}），平台期，学习率降至 {lr}。"
            else:
                reason += f"上轮结论 {history[-1][2]}（主指标 {history[-1][1]:.6f}），维持参数续训。"
        # Harness 大模型决策（llm.enabled 时尝试；不可用/校验失败自动回退规则）
        if (config.get("llm") or {}).get("enabled"):
            card = {"drift_type": "concept_drift" if degrade_streak > 0 else "no_drift",
                    "severity": drift.get("level", "normal"),
                    "suggested_action": "finetune", "expected_gain": 0.001}
            llm_plan = harness.decide_plan(card, drift, config)
            if llm_plan:
                p = llm_plan.get("params") or {}
                if "learning_rate" in p:
                    lr = float(p["learning_rate"])
                if "epochs" in p:
                    epochs = max(int(p["epochs"]), 1)
                decision.update(epochs=epochs, learning_rate=lr)
                reason = "大模型决策：" + llm_plan["reason"]

        logger.log("decide", detail={"decision": decision, "history": list(recent)},
                   decision_reason=reason)
        print(f"[platform] round-{r} 决策：{reason}")

        # ---- 环节 2：实验（续训，保留 Adam/标准化状态）----
        losses = engine.continue_round(model, train, decision["epochs"], decision["learning_rate"])
        cand_metrics = engine.platform_metrics(model, val)
        round_seconds = round(time.time() - t0, 3)
        total_seconds += round_seconds
        cost = {"training_seconds": round_seconds, "total_seconds": round(total_seconds, 1),
                "epochs": decision["epochs"], "steps_delta": len(losses)}

        # ---- 环节 3：门禁（方向感知）----
        g = gain(cand_metrics.get(primary, 0.0), best_metrics.get(primary, 0.0), primary)
        passed = g >= 0.0
        if passed:
            best_metrics = dict(cand_metrics)
            save_best(model)
            degrade_streak = 0
            conclusion = "采纳"
        else:
            model = Model.load(best_path)   # 回滚：恢复上一轮最优权重
            degrade_streak += 1
            conclusion = "回滚"
        verdict = (f"{primary} 候选 {cand_metrics.get(primary):.6f} vs 最优 {best_metrics.get(primary):.6f} "
                   f"（增益 {g:+.6f}）→ {conclusion}")
        logger.log("round", detail={"round": r, "metrics": cand_metrics,
                                    "gain": round(g, 6), "decision": decision},
                   metrics={"valid": cand_metrics}, resource=cost,
                   decision_reason=verdict, conclusion=conclusion,
                   artifact_path=best_path if passed else None)
        history.append((r, cand_metrics.get(primary, 0.0), conclusion, g))
        print(f"[platform] round-{r} {verdict}")

    # ---- 收尾 ----
    final = engine.platform_metrics(model, val)
    logger.log("finalize", detail={"profile": args.profile, "final_metrics": final,
                                   "baseline_metrics": base_metrics},
               metrics={"valid": final}, conclusion="收尾")
    print("=" * 60)
    print(f"[platform] 完成：{args.profile} 基线 {primary}={base_metrics.get(primary):.6f} "
          f"→ 最优 {primary}={final.get(primary):.6f}")
    print(f"[platform] 最优检查点：{best_path}")
    print(f"[platform] 记录：{logger.experiments_path(run_id)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
