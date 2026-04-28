import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/eval_samples/image_eval")
DEFAULT_DECISION_EVAL = DEFAULT_OUTPUT_DIR / "qwen3_vl_embedding_decision_eval.json"
DEFAULT_DECISION_REPORT = DEFAULT_OUTPUT_DIR / "qwen3_vl_embedding_decision_report.json"


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _normalize_mode(plan: dict) -> str:
    if not isinstance(plan, dict):
        return "none"
    return str(plan.get("mode") or "none").strip().lower() or "none"


def _failure_stage(reference: dict, actual_show: bool) -> str:
    gate = reference.get("image_gate") if isinstance(reference, dict) else {}
    plan = reference.get("image_render_plan") if isinstance(reference, dict) else {}
    if actual_show:
        return "none"
    stage = str(gate.get("failure_stage") or "").strip()
    if stage:
        return stage
    if gate.get("passed") and not actual_show:
        return "planner"
    if not reference.get("image_candidates"):
        return "retrieval"
    if plan:
        return "planner"
    return "gate"


def _evaluate(predictions: list[dict]):
    rows = []
    show_tp = show_fp = show_fn = show_tn = 0
    gate_pass_by_type = Counter()
    gate_total_by_type = Counter()
    gate_false_negative_by_type = Counter()
    planner_total = planner_valid = planner_fallback = 0
    multi_render_count = 0

    for sample in predictions:
        reference = sample.get("reference") or {}
        expected_action = str(sample.get("expected_action") or "").strip().lower()
        render_plan = reference.get("image_render_plan") or {}
        gate = reference.get("image_gate") or {}
        query_type = str(gate.get("query_type") or sample.get("query_type") or "unknown")
        actual_show = bool(render_plan.get("show_images"))
        render_mode = _normalize_mode(render_plan)
        render_item_count = len(render_plan.get("items") or []) if isinstance(render_plan, dict) else 0
        failure_stage = _failure_stage(reference, actual_show)
        planner_source = str(render_plan.get("planner_source") or "")

        gate_total_by_type[query_type] += 1
        if gate.get("passed"):
            gate_pass_by_type[query_type] += 1

        if expected_action == "show" and actual_show:
            show_tp += 1
        elif expected_action == "show" and not actual_show:
            show_fn += 1
            gate_false_negative_by_type[query_type] += 1
        elif expected_action == "hide" and actual_show:
            show_fp += 1
        else:
            show_tn += 1

        if render_mode == "single" and render_item_count > 1:
            multi_render_count += 1

        if gate.get("passed"):
            planner_total += 1
            if actual_show:
                planner_valid += 1
            if planner_source == "gate_fallback":
                planner_fallback += 1

        rows.append(
            {
                "qid": sample.get("qid"),
                "query_type": sample.get("query_type"),
                "expected_action": expected_action,
                "actual_show": actual_show,
                "render_mode": render_mode,
                "render_item_count": render_item_count,
                "gate_passed": bool(gate.get("passed")),
                "gate_reason": gate.get("reason_code"),
                "failure_stage": failure_stage,
                "planner_source": planner_source or None,
            }
        )

    evaluated = len(rows)
    report = {
        "evaluated_questions": evaluated,
        "show_tp": show_tp,
        "show_fp": show_fp,
        "show_fn": show_fn,
        "show_tn": show_tn,
        "show_precision": 0.0 if show_tp + show_fp == 0 else show_tp / (show_tp + show_fp),
        "show_recall": 0.0 if show_tp + show_fn == 0 else show_tp / (show_tp + show_fn),
        "false_show_rate": 0.0 if evaluated == 0 else show_fp / evaluated,
        "missed_show_rate": 0.0 if evaluated == 0 else show_fn / evaluated,
        "single_question_multi_render_rate": 0.0 if evaluated == 0 else multi_render_count / evaluated,
        "gate_pass_rate_by_type": {
            query_type: 0.0 if gate_total_by_type[query_type] == 0 else gate_pass_by_type[query_type] / gate_total_by_type[query_type]
            for query_type in sorted(gate_total_by_type)
        },
        "gate_false_negative_by_type": dict(sorted(gate_false_negative_by_type.items())),
        "planner_valid_plan_rate": 0.0 if planner_total == 0 else planner_valid / planner_total,
        "planner_fallback_rate": 0.0 if planner_total == 0 else planner_fallback / planner_total,
        "failure_stage_breakdown": dict(sorted(Counter(row["failure_stage"] for row in rows).items())),
    }
    return rows, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate image display decision quality from canary predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--decision-eval-output", default=str(DEFAULT_DECISION_EVAL))
    parser.add_argument("--decision-report-output", default=str(DEFAULT_DECISION_REPORT))
    args = parser.parse_args()

    predictions = _load_json(Path(args.predictions))
    rows, report = _evaluate(predictions)

    decision_eval_output = Path(args.decision_eval_output)
    decision_report_output = Path(args.decision_report_output)
    decision_eval_output.parent.mkdir(parents=True, exist_ok=True)
    decision_report_output.parent.mkdir(parents=True, exist_ok=True)
    decision_eval_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
