#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "image_assets"
IMAGE_EVAL_DIR = ROOT / "data" / "eval_samples" / "image_eval"

CANARY_PATH = IMAGE_EVAL_DIR / "image_search_canary_groundtruth_draft.json"
FULL_GROUNDTRUTH_PATH = IMAGE_EVAL_DIR / "image_search_full_groundtruth_v1.json"
FULL_MANIFEST_PATH = IMAGE_EVAL_DIR / "image_search_full_dataset_manifest_v1.json"
FULL_REPORT_PATH = IMAGE_EVAL_DIR / "image_search_full_build_report_v1.md"

EXCLUDED_DOC_UIDS: set[str] = set()

DIRTY_CAPTION_PATTERNS = [
    r"二维码",
    r"微信",
    r"QR code",
    r"license",
    r"Creative Commons",
    r"open access",
    r"Check for updates",
    r"作者简介",
    r"作者照片",
    r"个人简介",
    r"研究方向",
    r"硕士研究生",
    r"博士研究生",
    r"博士生导师",
    r"博士",
    r"教授",
    r"副教授",
    r"讲师",
    r"男[,，]",
    r"女[,，]",
    r"person seated",
    r"professional individual",
    r"Researcher in",
    r"Leadership\.",
    r"版权",
    r"在线交流",
    r"独家语音",
    r"研究必要性",
    r"未来趋势",
    r"摘要",
    r"关键词",
    r"责任编辑",
    r"/讲师",
]

NARRATIVE_CAPTION_PATTERNS = [
    r"探讨了",
    r"提出了",
    r"实验结果表明",
    r"应用与挑战",
    r"提高了",
    r"确保",
    r"优越性",
    r"通过与图",
    r"可以发现",
]

OVERVIEW_HINT_PATTERNS = [
    r"框图",
    r"框架",
    r"流程",
    r"结构",
    r"系统",
    r"示意",
    r"架构",
    r"网络",
    r"framework",
    r"overview",
    r"flowchart",
    r"taxonomy",
    r"architecture",
    r"topology",
    r"原理",
]

RESULT_HINT_PATTERNS = [
    r"结果",
    r"性能",
    r"曲线",
    r"混淆矩阵",
    r"accuracy",
    r"comparison",
    r"SINR",
    r"throughput",
    r"FID",
    r"波形",
    r"谱",
    r"频谱",
    r"对比",
]

FIGURE_LABEL_PATTERNS = [
    re.compile(r"(图\s*\d+(?:[-.]\d+)?)"),
    re.compile(r"(Fig\.?\s*\d+(?:[-.]\d+)?)", re.IGNORECASE),
    re.compile(r"(Figure\s*\d+(?:[-.]\d+)?)", re.IGNORECASE),
    re.compile(r"(FIGURE\s*\d+(?:[-.]\d+)?)", re.IGNORECASE),
]

SECONDARY_QUERY_TYPE_BY_DOC: dict[str, str] = {}

CANARY_SELECTION: dict[str, dict[str, str]] = {}

PRIMARY_FIGURE_OVERRIDES: dict[str, str] = {}

DOC_TOPIC_OVERRIDES: dict[str, str] = {}


@dataclass
class FigureRecord:
    doc_uid: str
    figure_key: str
    caption: str
    subject: str
    page: int
    figure_label: str
    single_score: int
    overview_score: int
    comparison_score: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


DIRTY_CAPTION_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in DIRTY_CAPTION_PATTERNS]
NARRATIVE_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in NARRATIVE_CAPTION_PATTERNS]
OVERVIEW_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in OVERVIEW_HINT_PATTERNS]
RESULT_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in RESULT_HINT_PATTERNS]


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page(figure_key: str) -> int:
    match = re.match(r"p(\d+)_", figure_key)
    if not match:
        raise ValueError(f"Cannot parse page from figure key: {figure_key}")
    return int(match.group(1))


def extract_figure_label(caption: str, figure_key: str) -> str:
    for pattern in FIGURE_LABEL_PATTERNS:
        match = pattern.search(caption)
        if match:
            return normalize_text(match.group(1))
    return figure_key


def strip_figure_label(caption: str) -> str:
    text = normalize_text(caption)
    text = re.sub(r"^(图\s*\d+(?:[-.]\d+)?[：:\s]*)", "", text)
    text = re.sub(r"^(Fig\.?\s*\d+(?:[-.]\d+)?[：:\s]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(Figure\s*\d+(?:[-.]\d+)?[：:\s]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(FIGURE\s*\d+(?:[-.]\d+)?[：:\s]*)", "", text)
    text = re.sub(r"^\([a-zA-Z]\)\s*", "", text)
    text = re.sub(r"^(显示了|展示了|给出了)\s*", "", text)
    return text.strip(" .:：;；")


def brief_subject(subject: str, limit: int = 28) -> str:
    text = normalize_text(subject)
    text = re.split(r"[，,。.;；:：]", text)[0].strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def finalize_sentence(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"。；", "；", text)
    text = re.sub(r"[。]{2,}", "。", text)
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[。.;；:：,，]+$", "", text)
    return text + "。"


def dedupe_key(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def is_dirty_caption(caption: str) -> bool:
    text = normalize_text(caption)
    if not text:
        return True
    if any(pattern.search(text) for pattern in DIRTY_CAPTION_REGEXES):
        return True
    has_visual_hint = bool(re.search(r"图\s*\d|Fig\.|Figure\s*\d|FIGURE\s*\d|框图|流程|结构|系统|示意|架构|网络|结果|性能|曲线|波形|谱", text, re.IGNORECASE))
    if not re.match(r"^(图\s*\d|Fig\.?\s*\d|Figure\s*\d|FIGURE\s*\d)", text, re.IGNORECASE):
        if re.search(r"如图\s*\d+\s*所示|图\s*\d+\s*给出|用图\s*\d+\s*描述", text):
            return True
    if len(text) > 48 and not has_visual_hint and any(pattern.search(text) for pattern in NARRATIVE_REGEXES):
        return True
    if len(text) > 120 and any(pattern.search(text) for pattern in NARRATIVE_REGEXES):
        return True
    return False


def compute_scores(caption: str, page: int, figure_key: str) -> tuple[int, int, int]:
    text = normalize_text(caption)
    label = extract_figure_label(text, figure_key)
    has_real_label = label != figure_key
    single = 0
    overview = 0
    comparison = 0

    if has_real_label:
        single += 18
        overview += 14
        comparison += 12

    if any(pattern.search(text) for pattern in OVERVIEW_REGEXES):
        single += 6
        overview += 16
        comparison += 5

    if any(pattern.search(text) for pattern in RESULT_REGEXES):
        single += 6
        overview += 4
        comparison += 12

    if 2 <= page <= 8:
        single += 2
        overview += 2
        comparison += 2

    text_len = len(text)
    if 8 <= text_len <= 120:
        single += 2
        overview += 2
        comparison += 2
    elif text_len < 6:
        single -= 8
        overview -= 8
        comparison -= 8

    if re.fullmatch(r"Figure\s*\d+\.?", text, re.IGNORECASE):
        single -= 10
        overview -= 8
        comparison -= 8

    if page == 1 and not has_real_label:
        single -= 4
        overview -= 4
        comparison -= 4

    return single, overview, comparison


def make_figure_record(doc_uid: str, figure_key: str, caption: str) -> FigureRecord:
    normalized = normalize_text(caption)
    subject = strip_figure_label(normalized)
    page = extract_page(figure_key)
    figure_label = extract_figure_label(normalized, figure_key)
    single_score, overview_score, comparison_score = compute_scores(normalized, page, figure_key)
    return FigureRecord(
        doc_uid=doc_uid,
        figure_key=figure_key,
        caption=normalized,
        subject=subject or normalized,
        page=page,
        figure_label=figure_label,
        single_score=single_score,
        overview_score=overview_score,
        comparison_score=comparison_score,
    )


def select_valid_figures(doc_uid: str, captions: dict[str, str]) -> tuple[list[FigureRecord], list[dict[str, Any]]]:
    valid: list[FigureRecord] = []
    filtered: list[dict[str, Any]] = []
    for figure_key, caption in captions.items():
        if is_dirty_caption(caption):
            filtered.append(
                {
                    "figure_key": figure_key,
                    "page": extract_page(figure_key),
                    "caption": normalize_text(caption),
                    "reason": "dirty_caption_rule",
                }
            )
            continue
        valid.append(make_figure_record(doc_uid, figure_key, caption))

    if not valid:
        raise RuntimeError(f"{doc_uid} has no valid figures after filtering")

    deduped: dict[str, FigureRecord] = {}
    for record in valid:
        key = dedupe_key(record.subject)
        current = deduped.get(key)
        if current is None or record.single_score > current.single_score:
            deduped[key] = record
    return list(deduped.values()), filtered


def human_doc_title(doc_uid: str) -> str:
    if doc_uid in DOC_TOPIC_OVERRIDES:
        return DOC_TOPIC_OVERRIDES[doc_uid]
    text = doc_uid.replace("_", " ")
    if re.search(r"[\u4e00-\u9fff]", text):
        text = re.sub(r"_[\u4e00-\u9fff]{2,4}$", "", doc_uid).replace("_", " ")
    return text.strip()


def topic_from_figure(record: FigureRecord) -> str:
    text = record.subject
    if re.search(r"框图|框架|流程|结构|系统|示意|架构|网络|framework|overview|flowchart|taxonomy|architecture", text, re.IGNORECASE):
        return "方法框架"
    if re.search(r"结果|性能|曲线|混淆矩阵|accuracy|comparison|SNR|SINR|throughput|FID", text, re.IGNORECASE):
        return "实验结果"
    return "核心内容"


def default_text_summary(doc_uid: str, figures: list[FigureRecord]) -> str:
    if doc_uid in DOC_TOPIC_OVERRIDES:
        return DOC_TOPIC_OVERRIDES[doc_uid]
    title = human_doc_title(doc_uid)
    subjects = [brief_subject(record.subject, limit=20) for record in figures[:2]]
    if re.match(r"^-?\d+$", doc_uid):
        return "、".join(subjects)
    if subjects:
        return f"{title}，并结合{ '、'.join(subjects) }等内容展开"
    return title


def build_ground_truth_image(record: FigureRecord, relevance: int, role: str, rationale: str) -> dict[str, Any]:
    return {
        "doc_uid": record.doc_uid,
        "figure_key": record.figure_key,
        "relevance": relevance,
        "role": role,
        "caption": record.caption,
        "rationale": rationale,
    }


def build_evidence(records: list[FigureRecord]) -> list[dict[str, Any]]:
    evidences: list[dict[str, Any]] = []
    for record in records:
        evidences.append(
            {
                "pdf_page": record.page,
                "figure_label": record.figure_label if record.figure_label != record.figure_key else "",
                "verification_note": f"已依据第{record.page}页图注与 figure_key={record.figure_key} 核对，该图对应“{brief_subject(record.subject, 32)}”。",
            }
        )
    return evidences


def build_text_evidence(record: FigureRecord) -> list[dict[str, Any]]:
    return [
        {
            "pdf_page": record.page,
            "figure_label": "",
            "verification_note": f"该题为纯文本控制题，文档中可参考的代表性图片位于第{record.page}页，但问题本身不要求出图。",
        }
    ]


def choose_primary_figure(doc_uid: str, figures: list[FigureRecord]) -> FigureRecord:
    if doc_uid in PRIMARY_FIGURE_OVERRIDES:
        for record in figures:
            if record.figure_key == PRIMARY_FIGURE_OVERRIDES[doc_uid]:
                return record
    return max(figures, key=lambda record: (record.single_score, record.overview_score, -record.page))


def choose_secondary_figures(doc_uid: str, query_type: str, figures: list[FigureRecord], primary: FigureRecord) -> list[FigureRecord]:
    others = [record for record in figures if record.figure_key != primary.figure_key]
    if query_type == "multi_image":
        ranked = sorted(
            figures,
            key=lambda record: (record.comparison_score, record.single_score, -record.page),
            reverse=True,
        )
        chosen: list[FigureRecord] = []
        for record in ranked:
            if not chosen:
                chosen.append(record)
                continue
            if dedupe_key(record.subject) == dedupe_key(chosen[0].subject):
                continue
            chosen.append(record)
            break
        if len(chosen) < 2:
            raise RuntimeError(f"{doc_uid} cannot support multi_image after filtering")
        return chosen

    if query_type == "broad_visual":
        ranked = sorted(
            figures,
            key=lambda record: (record.overview_score, record.single_score, -record.page),
            reverse=True,
        )
        chosen = [ranked[0]]
        for record in ranked[1:]:
            if dedupe_key(record.subject) != dedupe_key(chosen[0].subject):
                if record.overview_score >= 14 or record.comparison_score >= 18:
                    chosen.append(record)
                break
        return chosen

    return [primary]


def infer_secondary_query_type(figures: list[FigureRecord], primary: FigureRecord) -> str:
    others = [record for record in figures if record.figure_key != primary.figure_key]
    if len(others) >= 1:
        best_other = max(others, key=lambda record: (record.comparison_score, record.overview_score, record.single_score))
        if best_other.comparison_score >= 18:
            return "multi_image"
        if max(primary.overview_score, best_other.overview_score) >= 16:
            return "broad_visual"
    return "text_only"


def build_single_question(doc_uid: str, record: FigureRecord) -> dict[str, Any]:
    subject = brief_subject(record.subject, 36)
    reference = (
        f"对应图片是{record.figure_label}，它展示了{record.subject}"
        if record.figure_label != record.figure_key
        else f"对应图片展示了{record.subject}"
    )
    return {
        "question": f"在《{doc_uid}》这篇文档中，请展示“{subject}”对应的图片，并简要说明。",
        "query_type": "single_image",
        "question_type": "single_hop",
        "doc_uid": doc_uid,
        "expected_action": "show",
        "ground_truth_images": [
            build_ground_truth_image(record, relevance=2, role="primary", rationale="该图在清洗后候选中最适合作为单图主图。")
        ],
        "reference_answer": finalize_sentence(reference),
        "evidence": build_evidence([record]),
    }


def build_multi_question(doc_uid: str, records: list[FigureRecord]) -> dict[str, Any]:
    left, right = records[:2]
    left_brief = brief_subject(left.subject, 26)
    right_brief = brief_subject(right.subject, 26)
    answer = (
        f"“{left_brief}”主要展示{left.subject}；"
        f"“{right_brief}”主要展示{right.subject}"
        "。两张图从不同角度补充了该文的关键内容"
    )
    return {
        "question": f"请同时展示《{doc_uid}》中的“{left_brief}”和“{right_brief}”，并说明它们各自关注的内容。",
        "query_type": "multi_image",
        "question_type": "multi_hop",
        "doc_uid": doc_uid,
        "expected_action": "show",
        "ground_truth_images": [
            build_ground_truth_image(left, relevance=2, role="comparison", rationale="该图能代表第一个对比视角。"),
            build_ground_truth_image(right, relevance=2, role="comparison", rationale="该图能代表第二个对比视角。"),
        ],
        "reference_answer": finalize_sentence(answer),
        "evidence": build_evidence([left, right]),
    }


def build_broad_question(doc_uid: str, records: list[FigureRecord]) -> dict[str, Any]:
    primary = records[0]
    topic = topic_from_figure(primary)
    ground_truth_images = [
        build_ground_truth_image(primary, relevance=2, role="primary", rationale="该图最能概括文档的核心视觉信息。")
    ]
    evidence_records = [primary]
    if len(records) > 1:
        supporting = records[1]
        ground_truth_images.append(
            build_ground_truth_image(supporting, relevance=1, role="supporting", rationale="该图可作为辅助理解的补充图。")
        )
        evidence_records.append(supporting)
        answer = f"如果想快速理解这篇文档的{topic}，优先展示{primary.figure_label}最合适；再配合{supporting.figure_label}，可以补充{brief_subject(supporting.subject, 28)}这一层信息"
    else:
        answer = f"如果想快速理解这篇文档的{topic}，优先展示{primary.figure_label}最合适，因为它最集中地概括了{primary.subject}"
    return {
        "question": f"如果我想快速理解《{doc_uid}》的{topic}，最适合先展示哪张图？",
        "query_type": "broad_visual",
        "question_type": "multi_hop",
        "doc_uid": doc_uid,
        "expected_action": "show",
        "ground_truth_images": ground_truth_images,
        "reference_answer": finalize_sentence(answer),
        "evidence": build_evidence(evidence_records),
    }


def build_text_question(doc_uid: str, figures: list[FigureRecord]) -> dict[str, Any]:
    representative = figures[0]
    return {
        "question": f"《{doc_uid}》主要介绍了什么内容？",
        "query_type": "text_only",
        "question_type": "single_hop",
        "doc_uid": doc_uid,
        "expected_action": "hide",
        "ground_truth_images": [],
        "reference_answer": finalize_sentence(f"文档主要介绍了{default_text_summary(doc_uid, figures)}"),
        "evidence": build_text_evidence(representative),
    }


def load_canary_pool() -> dict[str, dict[str, dict[str, Any]]]:
    if not CANARY_SELECTION or not CANARY_PATH.is_file():
        return {}
    canary_data = load_json(CANARY_PATH)
    by_qid = {item["qid"]: item for item in canary_data}
    pool: dict[str, dict[str, dict[str, Any]]] = {}
    for doc_uid, selection in CANARY_SELECTION.items():
        pool[doc_uid] = {}
        for query_type, qid in selection.items():
            pool[doc_uid][query_type] = copy.deepcopy(by_qid[qid])
    return pool


def iter_doc_uids() -> list[str]:
    doc_uids = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name in {"eval_samples", "test"}:
            continue
        if path.name in EXCLUDED_DOC_UIDS:
            continue
        if not (path / "figures").is_dir():
            continue
        if not (path / "captions.json").exists():
            continue
        doc_uids.append(path.name)
    return doc_uids


def build_dataset() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    canary_pool = load_canary_pool()
    docs = iter_doc_uids()

    questions: list[dict[str, Any]] = []
    doc_entries: list[dict[str, Any]] = []
    reused_canary_count = 0

    for doc_uid in docs:
        captions = load_json(OUTPUT_ROOT / doc_uid / "captions.json")
        valid_figures, filtered_figures = select_valid_figures(doc_uid, captions)
        valid_figures = sorted(valid_figures, key=lambda record: (record.single_score, record.overview_score, record.comparison_score, -record.page), reverse=True)

        primary = choose_primary_figure(doc_uid, valid_figures)
        secondary_type = SECONDARY_QUERY_TYPE_BY_DOC.get(doc_uid) or infer_secondary_query_type(valid_figures, primary)
        selected_secondary_records = choose_secondary_figures(doc_uid, secondary_type, valid_figures, primary)

        doc_questions: list[dict[str, Any]] = []
        reused_qids: list[str] = []

        if doc_uid in canary_pool and "single_image" in canary_pool[doc_uid]:
            doc_questions.append(canary_pool[doc_uid]["single_image"])
            reused_qids.append(canary_pool[doc_uid]["single_image"]["qid"])
            reused_canary_count += 1
        else:
            doc_questions.append(build_single_question(doc_uid, primary))

        if doc_uid in canary_pool and secondary_type in canary_pool[doc_uid]:
            doc_questions.append(canary_pool[doc_uid][secondary_type])
            reused_qids.append(canary_pool[doc_uid][secondary_type]["qid"])
            reused_canary_count += 1
        else:
            if secondary_type == "multi_image":
                doc_questions.append(build_multi_question(doc_uid, selected_secondary_records))
            elif secondary_type == "broad_visual":
                doc_questions.append(build_broad_question(doc_uid, selected_secondary_records))
            elif secondary_type == "text_only":
                doc_questions.append(build_text_question(doc_uid, valid_figures))
            else:
                raise ValueError(f"Unsupported query type: {secondary_type}")

        questions.extend(doc_questions)
        doc_entries.append(
            {
                "doc_uid": doc_uid,
                "question_count": len(doc_questions),
                "query_types": [item["query_type"] for item in doc_questions],
                "reused_canary_qids": reused_qids,
                "valid_figure_count": len(valid_figures),
                "filtered_figure_count": len(filtered_figures),
                "selected_primary_figure_key": primary.figure_key,
                "selected_secondary_figure_keys": [record.figure_key for record in selected_secondary_records] if secondary_type != "text_only" else [],
                "filtered_figures": filtered_figures,
            }
        )

    for index, question in enumerate(questions, start=1):
        question["qid"] = f"img_full_{index:04d}"

    query_type_counter = Counter(item["query_type"] for item in questions)
    if any("evidence" not in question for question in questions):
        raise RuntimeError("Every question must include evidence")

    manifest = {
        "dataset_name": "image_search_full_groundtruth_v1",
        "version": "v1",
        "question_count": len(questions),
        "doc_count": len(docs),
        "query_type_distribution": dict(query_type_counter),
        "expected_action_distribution": dict(Counter(item["expected_action"] for item in questions)),
        "excluded_doc_uids": sorted(EXCLUDED_DOC_UIDS),
        "dirty_caption_patterns": DIRTY_CAPTION_PATTERNS,
        "reused_canary_question_count": reused_canary_count,
        "docs": doc_entries,
    }

    report = build_report(manifest)
    return questions, manifest, report


def build_report(manifest: dict[str, Any]) -> str:
    docs = manifest["docs"]
    total_filtered = sum(item["filtered_figure_count"] for item in docs)
    total_valid = sum(item["valid_figure_count"] for item in docs)
    reused_docs = [item["doc_uid"] for item in docs if item["reused_canary_qids"]]

    lines = [
        "# Image Search Full Groundtruth Build Report v1",
        "",
        "## Summary",
        f"- Covered docs: {manifest['doc_count']}",
        f"- Questions: {manifest['question_count']}",
        f"- Query types: {json.dumps(manifest['query_type_distribution'], ensure_ascii=False)}",
        f"- Expected actions: {json.dumps(manifest['expected_action_distribution'], ensure_ascii=False)}",
        f"- Reused canary questions: {manifest['reused_canary_question_count']}",
        f"- Valid figures kept after filtering: {total_valid}",
        f"- Figures filtered out as dirty captions: {total_filtered}",
        "",
        "## Canary Reuse",
        f"- Reused max two questions for each seeded doc, total docs involved: {len(reused_docs)}",
        f"- Seeded docs: {', '.join(reused_docs)}",
        "",
        "## Filtering Notes",
        "- Filtered caption patterns include QR codes, author bios, license/open-access pages, cover-style narrative blurbs, and other non-primary visuals.",
        "- Single-image questions prefer true figures such as flowcharts, system diagrams, structure charts, result plots, and comparison figures.",
        "- Text-only questions still include evidence, but ground truth images are intentionally empty and expected_action is `hide`.",
        "",
        "## Path Anomalies Observed",
        "- None recorded by default. Review the generated manifest if your local parser output uses custom directory names.",
        "",
        "## Top Filtered Docs",
    ]

    for entry in sorted(docs, key=lambda item: item["filtered_figure_count"], reverse=True)[:8]:
        lines.append(
            f"- {entry['doc_uid']}: filtered {entry['filtered_figure_count']} / kept {entry['valid_figure_count']}"
        )

    lines.extend(
        [
            "",
            "## Follow-up Risks",
            "- OCR-like narrative captions on page 1 are filtered heuristically, so future output regeneration may slightly change which figures survive.",
            "- If captions and manifests are split across sibling asset directories, the build relies on captions plus figure_key page parsing instead of a fully aligned manifest.",
            "- If upstream parsers emit hashed asset directory names, doc_uid will remain the directory name unless original document titles are available.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    questions, manifest, report = build_dataset()
    write_json(FULL_GROUNDTRUTH_PATH, questions)
    write_json(FULL_MANIFEST_PATH, manifest)
    FULL_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {FULL_GROUNDTRUTH_PATH}")
    print(f"Wrote {FULL_MANIFEST_PATH}")
    print(f"Wrote {FULL_REPORT_PATH}")


if __name__ == "__main__":
    main()
