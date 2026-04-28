import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/eval_samples/image_eval")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "qwen3_vl_embedding_predictions.json"
DEFAULT_EVAL = DEFAULT_OUTPUT_DIR / "qwen3_vl_embedding_predictions_eval.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "qwen3_vl_embedding_rag_eval_full_report.json"


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _normalize_qid(sample: dict) -> str:
    return str(sample.get("qid") or sample.get("id") or "").strip()


def _extract_image_candidates(sample: dict) -> list[dict]:
    if isinstance(sample.get("image_candidates"), list):
        return sample["image_candidates"]
    reference = sample.get("reference")
    if isinstance(reference, dict) and isinstance(reference.get("image_candidates"), list):
        return reference["image_candidates"]
    return []


def _extract_reference(sample: dict) -> dict:
    reference = sample.get("reference")
    return reference if isinstance(reference, dict) else {}


def _candidate_key(candidate: dict) -> tuple[str, str]:
    return (
        str(candidate.get("doc_name") or candidate.get("doc_uid") or "").strip(),
        str(candidate.get("figure_key") or "").strip(),
    )


def _label_key(label: dict) -> tuple[str, str]:
    return (
        str(label.get("doc_uid") or label.get("doc_name") or "").strip(),
        str(label.get("figure_key") or "").strip(),
    )


def _extract_labels(groundtruth_sample: dict, labels_field: str) -> list[dict]:
    labels = groundtruth_sample.get(labels_field) or []
    return labels if isinstance(labels, list) else []


def _evaluate(predictions: list[dict], groundtruth: list[dict], labels_field: str):
    gt_map = {
        _normalize_qid(sample): _extract_labels(sample, labels_field)
        for sample in groundtruth
        if _normalize_qid(sample)
    }
    per_question = []
    hit_counts = {1: 0, 3: 0, 5: 0}
    recall_totals = {1: 0.0, 3: 0.0, 5: 0.0}
    doc_hit_counts = {1: 0, 3: 0, 5: 0}
    role_hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    wrong_doc_count = 0
    evaluated = 0

    for sample in predictions:
        qid = _normalize_qid(sample)
        labels = gt_map.get(qid) or []
        if not qid or not labels:
            continue

        positives = {_label_key(label) for label in labels if _label_key(label) != ("", "")}
        positive_docs = {doc_name for doc_name, _ in positives}
        if not positives:
            continue

        candidates = _extract_image_candidates(sample)
        reference = _extract_reference(sample)
        gate = reference.get("image_gate") if isinstance(reference, dict) else {}
        ranked = [_candidate_key(candidate) for candidate in candidates if _candidate_key(candidate) != ("", "")]
        evaluated += 1
        if ranked and ranked[0][0] not in positive_docs:
            wrong_doc_count += 1

        first_hit_rank = None
        metrics = {"qid": qid, "positive_count": len(positives)}
        for k in (1, 3, 5):
            topk = ranked[:k]
            hits = [item for item in topk if item in positives]
            doc_hits = [item for item in topk if item[0] in positive_docs]
            role_hits = [
                candidate for candidate in candidates[:k]
                if str(candidate.get("semantic_role") or "").strip() in {"method_overview", "module_structure", "performance_curve", "comparison_result", "waveform_or_signal"}
            ]
            if hits:
                hit_counts[k] += 1
            if doc_hits:
                doc_hit_counts[k] += 1
            if role_hits:
                role_hit_counts[k] += 1
            recall_totals[k] += len(set(hits)) / len(positives)
            metrics[f"image_hit@{k}"] = 1 if hits else 0
            metrics[f"image_recall@{k}"] = len(set(hits)) / len(positives)
            metrics[f"doc_hit@{k}"] = 1 if doc_hits else 0
            metrics[f"role_aligned_hit@{k}"] = 1 if role_hits else 0

        for idx, item in enumerate(ranked, start=1):
            if item in positives:
                first_hit_rank = idx
                break
        if first_hit_rank:
            reciprocal_rank_sum += 1.0 / first_hit_rank
        metrics["image_mrr"] = 0.0 if first_hit_rank is None else 1.0 / first_hit_rank
        metrics["wrong_doc"] = 1 if ranked and ranked[0][0] not in positive_docs else 0
        metrics["failure_stage"] = str(gate.get("failure_stage") or "none")
        per_question.append(metrics)

    report = {
        "evaluated_questions": evaluated,
        "image_hit@1": 0.0 if evaluated == 0 else hit_counts[1] / evaluated,
        "image_hit@3": 0.0 if evaluated == 0 else hit_counts[3] / evaluated,
        "image_hit@5": 0.0 if evaluated == 0 else hit_counts[5] / evaluated,
        "image_recall@1": 0.0 if evaluated == 0 else recall_totals[1] / evaluated,
        "image_recall@3": 0.0 if evaluated == 0 else recall_totals[3] / evaluated,
        "image_recall@5": 0.0 if evaluated == 0 else recall_totals[5] / evaluated,
        "image_mrr": 0.0 if evaluated == 0 else reciprocal_rank_sum / evaluated,
        "doc_hit@1": 0.0 if evaluated == 0 else doc_hit_counts[1] / evaluated,
        "doc_hit@3": 0.0 if evaluated == 0 else doc_hit_counts[3] / evaluated,
        "doc_hit@5": 0.0 if evaluated == 0 else doc_hit_counts[5] / evaluated,
        "role_aligned_hit@1": 0.0 if evaluated == 0 else role_hit_counts[1] / evaluated,
        "role_aligned_hit@3": 0.0 if evaluated == 0 else role_hit_counts[3] / evaluated,
        "role_aligned_hit@5": 0.0 if evaluated == 0 else role_hit_counts[5] / evaluated,
        "wrong_doc_rate": 0.0 if evaluated == 0 else wrong_doc_count / evaluated,
    }
    return per_question, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate image_search predictions without affecting calibrated text reports.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--groundtruth", default="data/eval_samples/text_eval/electromagnetic_groundtruth.json")
    parser.add_argument("--labels-field", default="ground_truth_images")
    parser.add_argument("--predictions-eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    groundtruth_path = Path(args.groundtruth)
    predictions_eval_output = Path(args.predictions_eval_output)
    report_output = Path(args.report_output)

    predictions = _load_json(predictions_path)
    groundtruth = _load_json(groundtruth_path)
    per_question, report = _evaluate(predictions, groundtruth, args.labels_field)

    predictions_eval_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_eval_output.write_text(
        json.dumps(per_question, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
