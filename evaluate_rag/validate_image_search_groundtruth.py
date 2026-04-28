import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path("data/eval_samples/image_eval/image_search_full_groundtruth_v1.json")
DEFAULT_MANIFEST = Path("data/eval_samples/image_eval/image_search_full_dataset_manifest_v1.json")
DEFAULT_BUILD_REPORT = Path("data/eval_samples/image_eval/image_search_full_build_report_v1.md")
DEFAULT_ANNOTATION_PLAN = Path("data/eval_samples/text_eval/annotation_plan.csv")
DEFAULT_OUTPUT_ROOT = Path("output")
EXCLUDED_NO_FIGURES_DOC = "浅析数字信号处理技术的发展与应用_王广平"
PAGE_PATTERN = re.compile(r"p(\d+)_", flags=re.IGNORECASE)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def parse_annotation_plan(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    docs = set()
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        pdf_name = parts[1].strip()
        doc_uid = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
        if doc_uid != EXCLUDED_NO_FIGURES_DOC:
            docs.add(doc_uid)
    return docs


def page_from_figure_key(figure_key: str) -> int:
    match = PAGE_PATTERN.match(str(figure_key or ""))
    return int(match.group(1)) if match else 0


def append_validation_to_report(report_path: Path, summary: dict[str, Any]):
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "## 校验结果"
    section = [
        marker,
        f"- 通过项数量：{len(summary['passed_checks'])}",
        f"- 失败项数量：{len(summary['failed_checks'])}",
    ]
    if summary["failed_checks"]:
        for item in summary["failed_checks"]:
            section.append(f"- 失败：{item}")
    else:
        section.append("- 所有结构化硬校验均通过。")
    block = "\n".join(section) + "\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    report_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate full image search groundtruth and update manifest/report summaries.")
    parser.add_argument("--groundtruth", default=str(DEFAULT_DATASET))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--build-report", default=str(DEFAULT_BUILD_REPORT))
    parser.add_argument("--annotation-plan", default=str(DEFAULT_ANNOTATION_PLAN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    dataset_path = Path(args.groundtruth)
    manifest_path = Path(args.manifest)
    build_report_path = Path(args.build_report)
    annotation_plan_path = Path(args.annotation_plan)
    output_root = Path(args.output_root)

    dataset = load_json(dataset_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    allowed_docs = parse_annotation_plan(annotation_plan_path)

    failures = []
    passes = []
    qids = [item.get("qid") for item in dataset]
    if len(qids) == len(set(qids)):
        passes.append("qid_unique")
    else:
        failures.append("qid_unique")

    counts = Counter(item.get("query_type") for item in dataset)
    expected_counts = {
        "single_image": 49,
        "multi_image": 18,
        "broad_visual": 15,
        "text_only": 16,
    }
    if counts == expected_counts:
        passes.append("type_distribution")
    else:
        failures.append(f"type_distribution:{dict(counts)}")

    for item in dataset:
        doc_uid = item.get("doc_uid")
        query_type = item.get("query_type")
        expected_action = item.get("expected_action")
        images = item.get("ground_truth_images") or []
        evidence = item.get("evidence") or []
        if doc_uid not in allowed_docs:
            failures.append(f"doc_uid_not_allowed:{item.get('qid')}:{doc_uid}")
            continue
        if not str(item.get("reference_answer") or "").strip():
            failures.append(f"missing_reference_answer:{item.get('qid')}")
        if not evidence:
            failures.append(f"missing_evidence:{item.get('qid')}")
        if expected_action == "hide" and images == []:
            passes.append(f"hide_empty_images:{item.get('qid')}")
        elif expected_action == "hide":
            failures.append(f"hide_nonempty_images:{item.get('qid')}")
        if expected_action == "show" and images:
            passes.append(f"show_has_images:{item.get('qid')}")
        elif expected_action == "show":
            failures.append(f"show_missing_images:{item.get('qid')}")
        if query_type == "single_image" and len(images) == 1:
            passes.append(f"single_image_count:{item.get('qid')}")
        elif query_type == "single_image":
            failures.append(f"single_image_count:{item.get('qid')}")
        if query_type == "multi_image" and len(images) >= 2:
            passes.append(f"multi_image_count:{item.get('qid')}")
        elif query_type == "multi_image":
            failures.append(f"multi_image_count:{item.get('qid')}")
        for image in images:
            image_doc = image.get("doc_uid")
            figure_key = image.get("figure_key")
            if image_doc != doc_uid:
                failures.append(f"image_doc_mismatch:{item.get('qid')}:{figure_key}")
                continue
            captions_path = output_root / doc_uid / "captions.json"
            figures_path = output_root / doc_uid / "figures" / f"{figure_key}.jpg"
            if not captions_path.exists():
                failures.append(f"missing_captions:{item.get('qid')}:{doc_uid}")
                continue
            captions = load_json(captions_path)
            if figure_key not in captions:
                failures.append(f"missing_figure_key:{item.get('qid')}:{figure_key}")
            if not figures_path.exists():
                failures.append(f"missing_figure_file:{item.get('qid')}:{figure_key}")

    summary = {
        "status": "passed" if not failures else "failed",
        "passed_checks": passes,
        "failed_checks": failures,
        "question_count": len(dataset),
        "query_type_distribution": dict(counts),
    }

    manifest["validation_summary"] = summary
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    append_validation_to_report(build_report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
