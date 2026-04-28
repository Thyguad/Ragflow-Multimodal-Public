#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_ks(ks_str: str) -> List[int]:
    ks: List[int] = []
    for x in (ks_str or "").split(","):
        x = x.strip()
        if x:
            ks.append(int(x))
    ks = sorted(set(ks))
    if not ks:
        raise ValueError("--ks 不能为空，例如 1,3,5,10")
    if min(ks) <= 0:
        raise ValueError("--ks 的 k 必须为正整数")
    return ks


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s, flags=re.UNICODE)
    return s


def normalize_text_loose(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def dcg(relevances: Sequence[float]) -> float:
    score = 0.0
    for i, rel in enumerate(relevances, start=1):
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def evidence_position_dcg(evidence_hits: Sequence["EvidenceHit"], k: int) -> float:
    score = 0.0
    for hit in evidence_hits:
        if hit.best_position is None or hit.best_position > k:
            continue
        score += (2**hit.relevance - 1) / math.log2(hit.best_position + 1)
    return score


def ideal_evidence_position_dcg(evidences: Sequence["Evidence"]) -> float:
    # A single retrieved chunk can legitimately cover multiple evidence spans.
    # Use an evidence-level ideal DCG so packed evidence is not unfairly penalized.
    return sum((2**int(ev.relevance) - 1) for ev in evidences)


# Primary summary focuses on practical retrieval quality for downstream answering:
# coverage in the first few retrieved contexts matters more than exact top-1
# placement, because downstream answer generation typically consumes multiple
# chunks rather than only the first hit.
PRIMARY_RETRIEVAL_METRICS: Tuple[str, ...] = (
    "recall@3",
    "hitrate@3",
    "recall@5",
    "hitrate@5",
)

SECONDARY_STRICT_TOP1_METRICS: Tuple[str, ...] = (
    "recall@1",
    "ndcg@1",
)

PRIMARY_SUMMARY_FOCUS = "early_context_coverage_quality"
PRIMARY_SUMMARY_FORMULA = "mean(recall@3, hitrate@3, recall@5, hitrate@5)"
SAME_DOC_CONTEXT_NOTE = "analysis signal only; it does not receive direct retrieval credit"

MATCH_REASON_CONFIDENCE: Dict[str, float] = {
    "strict": 1.0,
    "answer_support": 0.935,
    # Same-document aggregation is useful for qualitative analysis, but it is
    # too weak to earn direct retrieval credit in the calibrated metrics.
    "same_doc_context": 0.0,
}

RETRIEVAL_SCORING_PROFILES: Dict[str, Dict[str, Any]] = {
    "primary_v2": {
        "default_ks": "1,3,5,10",
        "default_match_threshold": 0.82,
        "default_overlap_threshold": 0.75,
        "default_support_threshold": 0.70,
        "primary_metrics": ("recall@3", "hitrate@3", "recall@5", "hitrate@5"),
        "secondary_metrics": ("recall@1", "ndcg@1"),
        "focus": "early_context_coverage_quality",
        "formula": "mean(recall@3, hitrate@3, recall@5, hitrate@5)",
        "match_reason_confidence": {"strict": 1.0, "answer_support": 0.935, "same_doc_context": 0.0},
        "same_doc_context_note": "analysis signal only; it does not receive direct retrieval credit",
    },
    "legacy_strict_v1": {
        "default_ks": "1,3,5,8",
        "default_match_threshold": 0.82,
        "default_overlap_threshold": 0.75,
        "default_support_threshold": 0.70,
        "primary_metrics": ("recall@1", "recall@3", "hitrate@1", "hitrate@3", "ndcg@3", "mrr"),
        "secondary_metrics": ("recall@1", "ndcg@1"),
        "focus": "early_rank_quality",
        "formula": "mean(recall@1, recall@3, hitrate@1, hitrate@3, ndcg@3, mrr)",
        "match_reason_confidence": {"strict": 1.0, "answer_support": 1.0, "same_doc_context": 1.0},
        "same_doc_context_note": "same_doc_context receives full retrieval credit (legacy strict mode)",
    },
}


def doc_match(doc_uid: str, document_name: str) -> bool:
    return normalize_text(doc_uid) == normalize_text(document_name)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？；;!\?\n\r]+", str(text))
    return [p.strip() for p in parts if p.strip()]


def build_windows(text: str) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return [str(text)]

    windows: List[str] = []
    windows.extend(sents)
    for i in range(len(sents) - 1):
        windows.append(sents[i] + sents[i + 1])
    return windows


def char_overlap_ratio(a: str, b: str) -> float:
    sa = set(a)
    sb = set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def text_support_score(claim_text: str, answer_text: str) -> float:
    claim = normalize_text_loose(claim_text)
    answer = normalize_text_loose(answer_text)
    if not claim or not answer:
        return 0.0

    scores = [
        SequenceMatcher(None, claim, answer).ratio(),
        char_overlap_ratio(claim, answer),
    ]
    for win in build_windows(answer_text):
        chunk = normalize_text_loose(win)
        if not chunk:
            continue
        if claim in chunk:
            return 1.0
        scores.append(SequenceMatcher(None, claim, chunk).ratio())
        scores.append(char_overlap_ratio(claim, chunk))
    return max(scores) if scores else 0.0


def text_match(
    evidence_text: str,
    chunk_content: str,
    seq_threshold: float,
    overlap_threshold: float,
) -> bool:
    ev = normalize_text_loose(evidence_text)
    if not ev:
        return False

    best_seq = 0.0
    best_overlap = 0.0

    for win in build_windows(chunk_content):
        ch = normalize_text_loose(win)
        if not ch:
            continue

        if ev in ch:
            return True

        seq = SequenceMatcher(None, ev, ch).ratio()
        if seq > best_seq:
            best_seq = seq

        overlap = char_overlap_ratio(ev, ch)
        if overlap > best_overlap:
            best_overlap = overlap

    return (best_seq >= seq_threshold) or (best_overlap >= overlap_threshold)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    doc_uid: str
    evidence_text: str
    relevance: int


@dataclass
class EvidenceHit:
    evidence_id: str
    relevance: int
    best_position: Optional[int]
    best_chunk_id: Optional[str]
    best_document_name: Optional[str]
    best_reason: Optional[str]
    best_confidence: float


def validate_ground_truth(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("ground truth JSON 必须是数组")
    required_top_fields = ["qid", "question", "question_type", "ground_truth_evidences", "reference_answer"]
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"[ground_truth] item #{i} 必须是对象")
        for f in required_top_fields:
            if f not in item:
                raise ValueError(f"[ground_truth] item #{i} 缺少字段: {f}")
        if not isinstance(item["ground_truth_evidences"], list):
            raise ValueError(f"[ground_truth] item #{i} ground_truth_evidences 必须是数组")
        for j, ev in enumerate(item["ground_truth_evidences"]):
            if not isinstance(ev, dict):
                raise ValueError(f"[ground_truth] item #{i} evidence #{j} 必须是对象")
            for f in ["evidence_id", "doc_uid", "evidence_text", "relevance"]:
                if f not in ev:
                    raise ValueError(f"[ground_truth] item #{i} evidence #{j} 缺少字段: {f}")
    return items


def validate_predictions(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("predictions JSON 必须是数组")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"[predictions] item #{i} 必须是对象")
        if "qid" not in item:
            raise ValueError(f"[predictions] item #{i} 缺少字段: qid")
        chunks = prediction_chunks(item)
        if not isinstance(chunks, list):
            raise ValueError(f"[predictions] item #{i} retrieved_chunks/retrieval_chunks 必须是数组")
        for j, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                raise ValueError(f"[predictions] item #{i} chunk #{j} 必须是对象")
            if not chunk_document_name(ch):
                raise ValueError(f"[predictions] item #{i} chunk #{j} 缺少字段: document_name/docnm_kwd/doc_name/name")
            if chunk_content(ch) == "":
                raise ValueError(f"[predictions] item #{i} chunk #{j} 缺少字段: chunk_content/content/text")
    return items


def build_prediction_map(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in predictions:
        qid = item.get("qid")
        if qid is not None:
            out[str(qid)] = item
    return out


def to_evidences(gt_item: Dict[str, Any]) -> List[Evidence]:
    evidences: List[Evidence] = []
    for ev in gt_item.get("ground_truth_evidences", []):
        evidences.append(
            Evidence(
                evidence_id=str(ev.get("evidence_id", "")),
                doc_uid=str(ev.get("doc_uid", "")),
                evidence_text=str(ev.get("evidence_text", "")),
                relevance=int(ev.get("relevance", 1)),
            )
        )
    return evidences


def prediction_chunks(pred_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = pred_item.get("retrieved_chunks")
    if chunks is None:
        chunks = pred_item.get("retrieval_chunks")
    return list(chunks or [])


def chunk_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def chunk_document_name(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("document_name")
        or chunk.get("docnm_kwd")
        or chunk.get("doc_name")
        or chunk.get("name")
        or ""
    )


def chunk_content(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("chunk_content") or chunk.get("content") or chunk.get("text") or "")


def chunk_rank(chunk: Dict[str, Any], default_rank: int) -> float:
    return safe_float(chunk.get("rank", default_rank), default=float(default_rank))


def match_chunk(
    chunk: Dict[str, Any],
    evidences: Sequence[Evidence],
    seq_threshold: float,
    overlap_threshold: float,
) -> List[int]:
    doc_name = chunk_document_name(chunk)
    content = chunk_content(chunk)
    matched: List[int] = []
    for idx, ev in enumerate(evidences):
        if not doc_match(ev.doc_uid, doc_name):
            continue
        if text_match(
            ev.evidence_text,
            content,
            seq_threshold=seq_threshold,
            overlap_threshold=overlap_threshold,
        ):
            matched.append(idx)
    return matched


def match_evidence(
    chunk: Dict[str, Any],
    same_doc_context: str,
    evidences: Sequence[Evidence],
    gt_item: Dict[str, Any],
    generated_answer: str,
    seq_threshold: float,
    overlap_threshold: float,
    support_threshold: float,
) -> Dict[int, str]:
    doc_name = chunk_document_name(chunk)
    content = chunk_content(chunk)
    matched: Dict[int, str] = {}
    for idx, ev in enumerate(evidences):
        if not doc_match(ev.doc_uid, doc_name):
            continue

        if text_match(
            ev.evidence_text,
            content,
            seq_threshold=seq_threshold,
            overlap_threshold=overlap_threshold,
        ):
            matched[idx] = "strict"
            continue

        if same_doc_context and text_match(
            ev.evidence_text,
            same_doc_context,
            seq_threshold=seq_threshold,
            overlap_threshold=overlap_threshold,
        ):
            matched[idx] = "same_doc_context"
            continue

        evidence_score = text_support_score(ev.evidence_text, generated_answer)
        reference_score = text_support_score(str(gt_item.get("reference_answer", "")), generated_answer)
        if max(evidence_score, reference_score) >= support_threshold:
            matched[idx] = "answer_support"
    return matched


def evaluate_retrieval_one(
    gt_item: Dict[str, Any],
    pred_item: Dict[str, Any],
    ks: Sequence[int],
    seq_threshold: float,
    overlap_threshold: float,
    support_threshold: float,
) -> Dict[str, Any]:
    qid = str(gt_item.get("qid"))
    evidences = to_evidences(gt_item)
    total_evidence = len(evidences)

    retrieved = prediction_chunks(pred_item)
    retrieved = sorted(enumerate(retrieved, start=1), key=lambda x: chunk_rank(x[1], x[0]))
    retrieved = [chunk for _, chunk in retrieved]
    generated_answer = str(pred_item.get("generated_answer", "") or "")

    evidence_hits: List[EvidenceHit] = [
        EvidenceHit(
            evidence_id=ev.evidence_id,
            relevance=int(ev.relevance),
            best_position=None,
            best_chunk_id=None,
            best_document_name=None,
            best_reason=None,
            best_confidence=0.0,
        )
        for ev in evidences
    ]

    assigned_evidence: set[int] = set()
    matches: List[Dict[str, Any]] = []
    same_doc_contexts: Dict[str, str] = defaultdict(str)

    for pos, chunk in enumerate(retrieved, start=1):
        chunk_id_value = chunk_id(chunk)
        document_name = chunk_document_name(chunk)
        doc_key = normalize_text(document_name)
        current_content = chunk_content(chunk)
        same_doc_contexts[doc_key] = "\n".join(
            part for part in [same_doc_contexts.get(doc_key, ""), current_content] if part
        )
        matched_by_idx = match_evidence(
            chunk=chunk,
            same_doc_context=same_doc_contexts.get(doc_key, ""),
            evidences=evidences,
            gt_item=gt_item,
            generated_answer=generated_answer,
            seq_threshold=seq_threshold,
            overlap_threshold=overlap_threshold,
            support_threshold=support_threshold,
        )
        matched = list(matched_by_idx)
        new_matched = [i for i in matched if i not in assigned_evidence]

        for ev_idx in matched:
            reason = matched_by_idx[ev_idx]
            confidence = MATCH_REASON_CONFIDENCE.get(reason, 0.0)
            if confidence <= 0.0:
                continue
            hit = evidence_hits[ev_idx]
            if hit.best_position is None or pos < hit.best_position:
                hit.best_position = pos
                hit.best_chunk_id = chunk_id_value or None
                hit.best_document_name = document_name or None
                hit.best_reason = reason
                hit.best_confidence = confidence
            elif pos == hit.best_position and confidence > hit.best_confidence:
                hit.best_chunk_id = chunk_id_value or None
                hit.best_document_name = document_name or None
                hit.best_reason = reason
                hit.best_confidence = confidence

        if new_matched:
            assigned_evidence.update(new_matched)

        if matched:
            matches.append(
                {
                    "position": pos,
                    "chunk_id": chunk_id_value,
                    "document_name": document_name,
                    "matched_evidence_ids": [evidences[i].evidence_id for i in matched],
                    "matched_relevances": {evidences[i].evidence_id: int(evidences[i].relevance) for i in matched},
                    "match_reasons": {evidences[i].evidence_id: matched_by_idx[i] for i in matched},
                }
            )

    metrics: Dict[str, float] = {}
    for k in ks:
        credit_hits = [h for h in evidence_hits if (h.best_position is not None and h.best_position <= k)]
        recall_k = (
            sum(h.best_confidence for h in credit_hits) / total_evidence if total_evidence > 0 else 0.0
        )
        hitrate_k = max((h.best_confidence for h in credit_hits), default=0.0)

        dcg_k = sum(
            h.best_confidence * (2**h.relevance - 1) / math.log2(h.best_position + 1)
            for h in credit_hits
            if h.best_position is not None
        )
        idcg_k = ideal_evidence_position_dcg(evidences)
        ndcg_k = (dcg_k / idcg_k) if idcg_k > 0 else 0.0

        metrics[f"recall@{k}"] = round(recall_k, 6)
        metrics[f"hitrate@{k}"] = round(hitrate_k, 6)
        metrics[f"ndcg@{k}"] = round(ndcg_k, 6)

    metrics["mrr"] = round(
        max(
            ((h.best_confidence / h.best_position) for h in evidence_hits if h.best_position is not None),
            default=0.0,
        ),
        6,
    )

    missing_evidence_ids = [h.evidence_id for h in evidence_hits if h.best_position is None]

    return {
        "qid": qid,
        "question": gt_item.get("question", ""),
        "question_type": gt_item.get("question_type", "unknown"),
        "retrieval_metrics": metrics,
        "evidence_hits": [
            {
                "evidence_id": h.evidence_id,
                "relevance": h.relevance,
                "best_position": h.best_position,
                "best_chunk_id": h.best_chunk_id,
                "best_document_name": h.best_document_name,
                "best_reason": h.best_reason,
                "best_confidence": h.best_confidence,
            }
            for h in evidence_hits
        ],
        "missing_evidence_ids": missing_evidence_ids,
        "matches": matches,
    }


def average_retrieval_metrics(details: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, float]:
    metric_names = retrieval_metric_names(ks)

    out: Dict[str, float] = {}
    for name in metric_names:
        vals = [safe_float(d.get("retrieval_metrics", {}).get(name, 0.0)) for d in details]
        out[name] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return out


def retrieval_metric_names(ks: Sequence[int]) -> List[str]:
    metric_names: List[str] = []
    for k in ks:
        metric_names.extend([f"recall@{k}", f"hitrate@{k}", f"ndcg@{k}"])
    metric_names.append("mrr")
    return metric_names


def average_metric_dicts(metric_dicts: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, float]:
    metric_names = retrieval_metric_names(ks)

    out: Dict[str, float] = {}
    for name in metric_names:
        vals = [safe_float(metrics.get(name, 0.0)) for metrics in metric_dicts]
        out[name] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return out


def build_primary_retrieval_summary(metrics: Dict[str, float]) -> Dict[str, Any]:
    focus_metrics = {name: metrics[name] for name in PRIMARY_RETRIEVAL_METRICS if name in metrics}
    score = round(sum(focus_metrics.values()) / len(focus_metrics), 6) if focus_metrics else 0.0
    secondary_metrics = {
        name: metrics[name]
        for name in SECONDARY_STRICT_TOP1_METRICS
        if name in metrics
    }
    return {
        "focus": PRIMARY_SUMMARY_FOCUS,
        "aggregation": "balanced_by_question_type",
        "metric_weights": {name: 1.0 for name in focus_metrics},
        "formula": PRIMARY_SUMMARY_FORMULA,
        "metrics": focus_metrics,
        "score": score,
        "secondary_monitor": {
            "focus": "strict_top1",
            "metrics": secondary_metrics,
            "note": "Top-1 strict matching is reported for transparency but excluded from primary score.",
        },
    }


def evaluate_retrieval_all(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    ks: Sequence[int],
    seq_threshold: float,
    overlap_threshold: float,
    support_threshold: float,
) -> Dict[str, Any]:
    pred_map = build_prediction_map(predictions)
    details: List[Dict[str, Any]] = []
    for gt in ground_truth:
        qid = str(gt.get("qid"))
        pred = pred_map.get(qid, {"qid": qid, "retrieved_chunks": [], "generated_answer": ""})
        details.append(
            evaluate_retrieval_one(
                gt,
                pred,
                ks=ks,
                seq_threshold=seq_threshold,
                overlap_threshold=overlap_threshold,
                support_threshold=support_threshold,
            )
        )

    raw_overall = average_retrieval_metrics(details, ks=ks)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in details:
        grouped[str(d.get("question_type", "unknown"))].append(d)
    by_question_type = {k: average_retrieval_metrics(v, ks=ks) for k, v in grouped.items()}
    overall = average_metric_dicts(list(by_question_type.values()), ks=ks)
    primary_summary = build_primary_retrieval_summary(overall)

    return {
        "overall": overall,
        "macro_overall": overall,
        "raw_overall": raw_overall,
        "micro_overall": raw_overall,
        "primary_summary": primary_summary,
        "by_question_type": by_question_type,
        "details": details,
    }


def _doc_coverage_ratio(gt_item: Dict[str, Any], pred_item: Dict[str, Any], k: int) -> float:
    gt_docs = {
        normalize_text(ev.get("doc_uid", ""))
        for ev in gt_item.get("ground_truth_evidences", [])
        if normalize_text(ev.get("doc_uid", ""))
    }
    if not gt_docs:
        return 0.0

    pred_docs = {
        normalize_text(chunk_document_name(ch))
        for ch in prediction_chunks(pred_item)[:k]
        if normalize_text(chunk_document_name(ch))
    }
    return len(gt_docs & pred_docs) / len(gt_docs)


def evaluate_structural_retrieval_signals(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
) -> Dict[str, Any]:
    pred_map = build_prediction_map(predictions)
    gt_doc_cov_at_3: List[float] = []
    gt_doc_cov_at_5: List[float] = []
    multi_hop_multi_doc_rate_at_5: List[float] = []

    for gt_item in ground_truth:
        qid = str(gt_item.get("qid"))
        pred_item = pred_map.get(qid, {"qid": qid, "retrieved_chunks": []})
        chunks = prediction_chunks(pred_item)
        top5_docs = [
            normalize_text(chunk_document_name(ch))
            for ch in chunks[:5]
            if normalize_text(chunk_document_name(ch))
        ]

        gt_doc_cov_at_3.append(_doc_coverage_ratio(gt_item, pred_item, 3))
        gt_doc_cov_at_5.append(_doc_coverage_ratio(gt_item, pred_item, 5))

        if str(gt_item.get("question_type", "")) == "multi_hop":
            multi_hop_multi_doc_rate_at_5.append(1.0 if len(set(top5_docs)) >= 2 else 0.0)

    by_question_type = retrieval.get("by_question_type", {})
    type_balance_metrics: Dict[str, float] = {}
    if "single_hop" in by_question_type and "multi_hop" in by_question_type:
        for metric_name in PRIMARY_RETRIEVAL_METRICS:
            if metric_name not in by_question_type["single_hop"] or metric_name not in by_question_type["multi_hop"]:
                continue
            delta = abs(
                safe_float(by_question_type["single_hop"].get(metric_name, 0.0))
                - safe_float(by_question_type["multi_hop"].get(metric_name, 0.0))
            )
            type_balance_metrics[metric_name] = round(1.0 - delta, 6)

    type_balance_score = (
        round(sum(type_balance_metrics.values()) / len(type_balance_metrics), 6)
        if type_balance_metrics
        else 0.0
    )
    structural_robustness_score = round(
        (
            (sum(gt_doc_cov_at_3) / len(gt_doc_cov_at_3) if gt_doc_cov_at_3 else 0.0)
            + (sum(gt_doc_cov_at_5) / len(gt_doc_cov_at_5) if gt_doc_cov_at_5 else 0.0)
            + (
                sum(multi_hop_multi_doc_rate_at_5) / len(multi_hop_multi_doc_rate_at_5)
                if multi_hop_multi_doc_rate_at_5
                else 0.0
            )
            + type_balance_score
        ) / 4.0,
        6,
    )

    return {
        "summary": {
            "score": structural_robustness_score,
            "formula": "mean(gt_doc_coverage@3, gt_doc_coverage@5, multi_hop_multi_doc_rate@5, type_balance.score)",
            "focus": "structure_preservation_and_multi_hop_readiness",
        },
        "gt_doc_coverage@3": round(sum(gt_doc_cov_at_3) / len(gt_doc_cov_at_3), 6) if gt_doc_cov_at_3 else 0.0,
        "gt_doc_coverage@5": round(sum(gt_doc_cov_at_5) / len(gt_doc_cov_at_5), 6) if gt_doc_cov_at_5 else 0.0,
        "multi_hop_multi_doc_rate@5": round(
            sum(multi_hop_multi_doc_rate_at_5) / len(multi_hop_multi_doc_rate_at_5),
            6,
        ) if multi_hop_multi_doc_rate_at_5 else 0.0,
        "type_balance": {
            "score": type_balance_score,
            "metrics": type_balance_metrics,
            "note": "Higher means the system behaves more consistently across single-hop and multi-hop questions.",
        },
    }


def build_ragas_dataset_rows(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    max_contexts: Optional[int],
) -> List[Dict[str, Any]]:
    pred_map = build_prediction_map(predictions)
    rows: List[Dict[str, Any]] = []
    for gt in ground_truth:
        qid = str(gt.get("qid"))
        pred = pred_map.get(qid, {})
        retrieved = sorted(
            enumerate(prediction_chunks(pred), start=1),
            key=lambda x: chunk_rank(x[1], x[0]),
        )
        retrieved = [chunk for _, chunk in retrieved]
        if max_contexts is not None:
            retrieved = retrieved[: int(max_contexts)]
        rows.append(
            {
                "qid": qid,
                "question_type": gt.get("question_type", "unknown"),
                "question": gt.get("question", "") or "",
                "answer": pred.get("generated_answer", "") or "",
                "contexts": [c.get("chunk_content", "") or "" for c in retrieved],
                "ground_truth": gt.get("reference_answer", "") or "",
            }
        )
    return rows


def normalize_openai_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = str(url).strip().rstrip("/")
    for suffix in ("/v1/embeddings", "/embeddings", "/v1/chat/completions", "/chat/completions"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            u = u.rstrip("/")
            break
    return u


class EmbeddingsAdapter:
    def __init__(self, inner: Any):
        self._inner = inner

    def embed_query(self, text: str) -> List[float]:
        return list(self._inner.embed_text(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self._inner.embed_texts(texts)]

    async def aembed_query(self, text: str) -> List[float]:
        return list(await self._inner.aembed_text(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        out = await self._inner.aembed_texts(texts)
        return [list(v) for v in out]


def make_openai_ragas_runtime(
    llm_api_key: Optional[str],
    llm_base_url: Optional[str],
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
    llm_model: str,
    embedding_model: str,
    llm_timeout: Optional[float],
    embedding_timeout: Optional[float],
    llm_max_retries: int,
    embedding_max_retries: int,
) -> Tuple[Any, Any]:
    import inspect

    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise RuntimeError("未找到 LLM API KEY；请设置 OPENAI_API_KEY 或通过参数传入 --llm-api-key")
    llm_client_kwargs: Dict[str, Any] = {"api_key": llm_api_key}
    llm_base_url = normalize_openai_base_url(llm_base_url)
    if llm_base_url:
        llm_client_kwargs["base_url"] = llm_base_url
    if llm_timeout is not None:
        llm_client_kwargs["timeout"] = float(llm_timeout)
    llm_client_kwargs["max_retries"] = int(llm_max_retries)

    embedding_api_key = embedding_api_key if embedding_api_key is not None else llm_api_key
    embedding_client_kwargs: Dict[str, Any] = {"api_key": embedding_api_key}
    embedding_base_url = normalize_openai_base_url(embedding_base_url)
    if embedding_base_url:
        embedding_client_kwargs["base_url"] = embedding_base_url
    if embedding_timeout is not None:
        embedding_client_kwargs["timeout"] = float(embedding_timeout)
    embedding_client_kwargs["max_retries"] = int(embedding_max_retries)

    if "client" not in inspect.signature(llm_factory).parameters:
        from langchain_openai import ChatOpenAI
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper

        class OpenAICompatibleEmbeddings:
            def __init__(self) -> None:
                self.client = OpenAI(**embedding_client_kwargs)
                self.async_client = AsyncOpenAI(**embedding_client_kwargs)

            @staticmethod
            def _coerce_text(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    return "\n".join(OpenAICompatibleEmbeddings._coerce_text(v) for v in value)
                if isinstance(value, dict):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)

            @classmethod
            def _coerce_texts(cls, values: Any) -> List[str]:
                if isinstance(values, str):
                    return [values]
                return [cls._coerce_text(v) for v in list(values or [])]

            def embed_query(self, text: str) -> List[float]:
                return self.embed_documents([text])[0]

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                normalized = self._coerce_texts(texts)
                res = self.client.embeddings.create(model=embedding_model, input=normalized)
                return [list(item.embedding) for item in res.data]

            async def aembed_query(self, text: str) -> List[float]:
                out = await self.aembed_documents([text])
                return out[0]

            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                normalized = self._coerce_texts(texts)
                res = await self.async_client.embeddings.create(model=embedding_model, input=normalized)
                return [list(item.embedding) for item in res.data]

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                temperature=0,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(OpenAICompatibleEmbeddings())
        return llm, embeddings

    llm_client = AsyncOpenAI(**llm_client_kwargs)
    llm = llm_factory(llm_model, client=llm_client)
    embedding_client = AsyncOpenAI(**embedding_client_kwargs)
    embeddings = embedding_factory("openai", model=embedding_model, client=embedding_client)
    return llm, EmbeddingsAdapter(embeddings)


def evaluate_generation_with_ragas(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    llm_model: str,
    embedding_model: str,
    llm_api_key: Optional[str],
    llm_base_url: Optional[str],
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
    max_contexts: Optional[int],
    raise_exceptions: bool,
    batch_size: Optional[int],
    llm_timeout: Optional[float],
    embedding_timeout: Optional[float],
    llm_max_retries: int,
    embedding_max_retries: int,
) -> Dict[str, Any]:
    import inspect

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows = build_ragas_dataset_rows(ground_truth, predictions, max_contexts=max_contexts)
    try:
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError("未找到 datasets 依赖；请先安装 ragas[datasets] 或 datasets") from e

    dataset = Dataset.from_list(rows)
    llm, embeddings = make_openai_ragas_runtime(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        llm_timeout=llm_timeout,
        embedding_timeout=embedding_timeout,
        llm_max_retries=llm_max_retries,
        embedding_max_retries=embedding_max_retries,
    )

    metrics = [
        copy.deepcopy(faithfulness),
        copy.deepcopy(answer_relevancy),
        copy.deepcopy(context_precision),
        copy.deepcopy(context_recall),
    ]

    evaluate_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": llm,
        "embeddings": embeddings,
        "raise_exceptions": bool(raise_exceptions),
    }
    evaluate_params = inspect.signature(evaluate).parameters
    if "batch_size" in evaluate_params:
        evaluate_kwargs["batch_size"] = batch_size
    elif "run_config" in evaluate_params and batch_size:
        from ragas.run_config import RunConfig

        timeouts = [v for v in [llm_timeout, embedding_timeout] if v is not None]
        evaluate_kwargs["run_config"] = RunConfig(
            timeout=max(timeouts) if timeouts else 180,
            max_workers=int(batch_size),
        )
    result = evaluate(**evaluate_kwargs)

    metric_names = [m.name for m in metrics]

    overall_scores: Dict[str, Optional[float]] = {}
    repr_dict = getattr(result, "_repr_dict", None)
    if isinstance(repr_dict, dict):
        for k, v in repr_dict.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                overall_scores[str(k)] = None
            else:
                overall_scores[str(k)] = round(safe_float(v, default=float("nan")), 6)
    else:
        buckets: Dict[str, List[float]] = {n: [] for n in metric_names}
        for row_score in (getattr(result, "scores", None) or []):
            if not isinstance(row_score, dict):
                continue
            for n in metric_names:
                v = row_score.get(n)
                vf = safe_float(v, default=float("nan"))
                if not math.isnan(vf):
                    buckets[n].append(vf)
        for n, vals in buckets.items():
            overall_scores[n] = round(sum(vals) / len(vals), 6) if vals else None

    per_row: List[Dict[str, Any]] = []
    scores_list = getattr(result, "scores", None) or []
    for i, score_dict in enumerate(scores_list):
        base = rows[i] if i < len(rows) else {}
        if not isinstance(score_dict, dict):
            continue
        gen_metrics: Dict[str, Optional[float]] = {}
        for n in metric_names:
            v = score_dict.get(n)
            vf = safe_float(v, default=float("nan"))
            gen_metrics[n] = None if math.isnan(vf) else round(vf, 6)
        per_row.append(
            {
                "qid": base.get("qid"),
                "question_type": base.get("question_type", "unknown"),
                "generation_metrics": gen_metrics,
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_row:
        grouped[str(r.get("question_type", "unknown"))].append(r)
    by_question_type: Dict[str, Dict[str, Optional[float]]] = {}
    for qtype, items in grouped.items():
        q_scores: Dict[str, Optional[float]] = {}
        for m in metrics:
            vals = [it["generation_metrics"].get(m.name) for it in items if it["generation_metrics"].get(m.name) is not None]
            q_scores[m.name] = round(sum(vals) / len(vals), 6) if vals else None
        by_question_type[qtype] = q_scores

    return {
        "provider": "openai",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "overall": overall_scores,
        "by_question_type": by_question_type,
        "details": per_row,
    }


def merge_generation_into_retrieval_details(
    retrieval: Dict[str, Any],
    generation: Dict[str, Any],
) -> None:
    gen_map = {d.get("qid"): d.get("generation_metrics", {}) for d in generation.get("details", [])}
    for d in retrieval.get("details", []):
        qid = d.get("qid")
        if qid in gen_map:
            d["generation_metrics"] = gen_map[qid]


def apply_retrieval_scoring_profile(profile_name: str) -> Dict[str, Any]:
    global PRIMARY_RETRIEVAL_METRICS
    global SECONDARY_STRICT_TOP1_METRICS
    global PRIMARY_SUMMARY_FOCUS
    global PRIMARY_SUMMARY_FORMULA
    global SAME_DOC_CONTEXT_NOTE
    global MATCH_REASON_CONFIDENCE

    profile = RETRIEVAL_SCORING_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown retrieval scoring profile: {profile_name}")

    PRIMARY_RETRIEVAL_METRICS = tuple(profile.get("primary_metrics", PRIMARY_RETRIEVAL_METRICS))
    SECONDARY_STRICT_TOP1_METRICS = tuple(profile.get("secondary_metrics", SECONDARY_STRICT_TOP1_METRICS))
    PRIMARY_SUMMARY_FOCUS = str(profile.get("focus", PRIMARY_SUMMARY_FOCUS))
    PRIMARY_SUMMARY_FORMULA = str(profile.get("formula", PRIMARY_SUMMARY_FORMULA))
    SAME_DOC_CONTEXT_NOTE = str(profile.get("same_doc_context_note", SAME_DOC_CONTEXT_NOTE))

    MATCH_REASON_CONFIDENCE.clear()
    MATCH_REASON_CONFIDENCE.update(profile.get("match_reason_confidence", {}))
    return profile


def main() -> None:
    p = argparse.ArgumentParser(description="RAG ???????????? ragas ?????")
    p.add_argument("--ground-truth", required=True, help="ground truth JSON ??")
    p.add_argument("--predictions", required=True, help="predictions JSON ??")
    p.add_argument("--output", required=True, help="???? JSON ??")
    p.add_argument(
        "--retrieval-scoring-profile",
        default="primary_v2",
        choices=sorted(RETRIEVAL_SCORING_PROFILES.keys()),
        help="retrieval scoring profile",
    )
    p.add_argument("--ks", default=None, help="??? top-k??? 1,3,5,10")
    p.add_argument("--match-threshold", type=float, default=None, help="evidence ? chunk ??????????? 0-1")
    p.add_argument("--overlap-threshold", type=float, default=None, help="evidence ?????????? 0-1")
    p.add_argument(
        "--support-threshold",
        type=float,
        default=None,
        help="evidence/reference answer ? generated_answer ????? 0-1",
    )

    p.add_argument("--enable-generation", action="store_true", help="???? ragas ????")
    p.add_argument("--generation-llm-model", default="gpt-4o-mini", help="ragas ??? LLM ???")
    p.add_argument("--generation-embedding-model", default="text-embedding-3-small", help="ragas ??? embedding ???")
    p.add_argument("--llm-api-key", default=None, help="LLM API Key")
    p.add_argument("--llm-base-url", default=None, help="LLM ? OpenAI ?? base_url")
    p.add_argument("--embedding-api-key", default=None, help="Embedding API Key")
    p.add_argument("--embedding-base-url", default=None, help="Embedding ? OpenAI ?? base_url")
    p.add_argument("--llm-timeout", type=float, default=120.0, help="LLM ???????")
    p.add_argument("--embedding-timeout", type=float, default=120.0, help="Embedding ???????")
    p.add_argument("--llm-max-retries", type=int, default=2, help="LLM ??????")
    p.add_argument("--embedding-max-retries", type=int, default=2, help="Embedding ??????")
    p.add_argument("--generation-api-key", default=None, help="????????? --llm-api-key")
    p.add_argument("--generation-base-url", default=None, help="????????? --llm-base-url")
    p.add_argument("--generation-max-contexts", type=int, default=None, help="ragas ??? top-N contexts")
    p.add_argument("--ragas-raise-exceptions", action="store_true", help="ragas ????????")
    p.add_argument("--ragas-batch-size", type=int, default=None, help="ragas ???")

    args = p.parse_args()

    if not os.path.exists(args.ground_truth):
        raise FileNotFoundError(f"ground truth ?????: {args.ground_truth}")
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"predictions ?????: {args.predictions}")

    profile = apply_retrieval_scoring_profile(args.retrieval_scoring_profile)
    ks = parse_ks(args.ks or str(profile["default_ks"]))
    match_threshold = (
        float(args.match_threshold)
        if args.match_threshold is not None
        else float(profile["default_match_threshold"])
    )
    overlap_threshold = (
        float(args.overlap_threshold)
        if args.overlap_threshold is not None
        else float(profile["default_overlap_threshold"])
    )
    support_threshold = (
        float(args.support_threshold)
        if args.support_threshold is not None
        else float(profile["default_support_threshold"])
    )

    if not (0.0 <= match_threshold <= 1.0):
        raise ValueError("--match-threshold ??? [0, 1] ???")
    if not (0.0 <= overlap_threshold <= 1.0):
        raise ValueError("--overlap-threshold ??? [0, 1] ???")
    if not (0.0 <= support_threshold <= 1.0):
        raise ValueError("--support-threshold ??? [0, 1] ???")

    ground_truth = validate_ground_truth(load_json(args.ground_truth))
    predictions = validate_predictions(load_json(args.predictions))

    retrieval = evaluate_retrieval_all(
        ground_truth=ground_truth,
        predictions=predictions,
        ks=ks,
        seq_threshold=match_threshold,
        overlap_threshold=overlap_threshold,
        support_threshold=support_threshold,
    )
    structural_signals = evaluate_structural_retrieval_signals(
        ground_truth=ground_truth,
        predictions=predictions,
        retrieval=retrieval,
    )

    report: Dict[str, Any] = {
        "config": {
            "retrieval_scoring_profile": args.retrieval_scoring_profile,
            "ks": ks,
            "match_threshold": match_threshold,
            "overlap_threshold": overlap_threshold,
            "support_threshold": support_threshold,
            "match_reason_confidence": MATCH_REASON_CONFIDENCE,
            "aggregation_notes": {
                "overall": "balanced macro average across question_type groups",
                "raw_overall": "micro average across all questions",
                "primary_summary": "headline early-rank quality score derived from overall",
                "same_doc_context": SAME_DOC_CONTEXT_NOTE,
            },
        },
        "retrieval": retrieval,
        "retrieval_structural_signals": structural_signals,
        "headline_scores": {
            "retrieval_quality": retrieval.get("primary_summary", {}),
            "structural_robustness": structural_signals.get("summary", {}),
        },
    }

    if args.enable_generation:
        llm_api_key = args.llm_api_key or args.generation_api_key
        llm_base_url = args.llm_base_url or args.generation_base_url
        embedding_api_key = args.embedding_api_key
        embedding_base_url = args.embedding_base_url
        if llm_base_url and (str(llm_base_url).strip().rstrip("/").endswith("/embeddings") or str(llm_base_url).strip().rstrip("/").endswith("/v1/embeddings")) and not embedding_base_url:
            embedding_base_url = llm_base_url
            llm_base_url = None
        generation = evaluate_generation_with_ragas(
            ground_truth=ground_truth,
            predictions=predictions,
            llm_model=args.generation_llm_model,
            embedding_model=args.generation_embedding_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            max_contexts=args.generation_max_contexts,
            raise_exceptions=bool(args.ragas_raise_exceptions),
            batch_size=args.ragas_batch_size,
            llm_timeout=args.llm_timeout,
            embedding_timeout=args.embedding_timeout,
            llm_max_retries=args.llm_max_retries,
            embedding_max_retries=args.embedding_max_retries,
        )
        report["generation"] = generation
        merge_generation_into_retrieval_details(report["retrieval"], generation)

    dump_json(report, args.output)

    print("\n=== Retrieval Overall ===")
    print(json.dumps(report["retrieval"]["overall"], ensure_ascii=False, indent=2))
    print("\n=== Retrieval By Question Type ===")
    print(json.dumps(report["retrieval"]["by_question_type"], ensure_ascii=False, indent=2))
    print("\n=== Retrieval Structural Signals ===")
    print(json.dumps(report.get("retrieval_structural_signals", {}), ensure_ascii=False, indent=2))
    print("\n=== Headline Scores ===")
    print(json.dumps(report.get("headline_scores", {}), ensure_ascii=False, indent=2))
    if args.enable_generation:
        print("\n=== Generation Overall ===")
        print(json.dumps(report.get("generation", {}).get("overall", {}), ensure_ascii=False, indent=2))
        print("\n=== Generation By Question Type ===")
        print(json.dumps(report.get("generation", {}).get("by_question_type", {}), ensure_ascii=False, indent=2))
    print(f"\n[OK] Report written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
