import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag.image_asset_utils import (
    DEFAULT_RUN_ID,
    IMAGE_FUSION_CAPTION_WEIGHT,
    IMAGE_FUSION_IMAGE_WEIGHT,
    IMAGE_INDEX_VERSION,
    build_asset_relpath,
    captions_hash,
    derive_run_id,
    get_doc_asset_dir,
    get_doc_figures_dir,
    get_doc_image_index_manifest_path,
    get_doc_image_index_vectors_path,
    get_figure_path,
    load_captions_dict,
    load_json_file,
    write_json_file,
)
from rag.multimodal_embedding_client import Qwen3VLEmbeddingClient


LOGGER = logging.getLogger(__name__)
IMAGE_INDEX_SOURCE = "image_index"
PER_DOC_RAW_TOP_K = 6
GLOBAL_CANDIDATE_TOP_K = 6
SAME_ROLE_DEDUPE_GAP = 0.03
METHOD_OVERVIEW_KEYWORDS = (
    "整体结构",
    "总体结构",
    "总结构",
    "结构图",
    "总览图",
    "方法结构",
    "方法框架",
    "框架图",
    "流程图",
    "框图",
    "系统框图",
    "总体框架",
    "整体方法",
    "核心方法",
    "流程",
    "方案",
    "示意图",
    "block diagram",
    "architecture",
    "framework",
    "pipeline",
    "overview",
    "flow chart",
    "scheme",
    "methodology",
    "based identification",
    "identification framework",
)
MODULE_STRUCTURE_KEYWORDS = (
    "module",
    "modules",
    "network",
    "subnetwork",
    "feature extraction network",
    "residual structure",
    "residual network",
    "inception module",
    "squeeze and excitation",
    "模块",
    "子模块",
    "网络",
    "网络结构",
    "特征提取网络",
    "残差结构",
    "残差网络",
    "卷积模块",
)
WAVEFORM_SIGNAL_KEYWORDS = (
    "waveform",
    "waveforms",
    "received signal",
    "valid segment",
    "delay estimation",
    "demodulation",
    "signal waveform",
    "波形",
    "时域",
    "频域",
    "输出波形",
    "接收信号",
    "有效段",
    "延时估计",
    "解调",
)
PERFORMANCE_OVERVIEW_KEYWORDS = (
    "识别性能",
    "性能",
    "效果",
    "准确率",
    "性能表现",
    "result",
    "results",
    "identification results",
    "results under different",
    "performance",
    "accuracy",
    "curve",
    "confusion matrix",
    "error matrix",
    "识别结果",
    "性能对比",
)
COMPARE_KEYWORDS = ("compare", "comparison", "对比", "比较", "区别", "difference", "versus", "vs", "同时展示")
CORE_METHOD_KEYWORDS = (
    "核心方法",
    "核心流程",
    "整体方法",
    "总体方法",
    "方法概括",
    "总结",
    "overview",
    "overall method",
    "core method",
    "main method",
)
MULTI_IMAGE_MARKERS = ("同时展示", "分别展示", "多张图", "几张图")
PAIR_REQUEST_MARKERS = ("两张图", "2张图", "2 张图", "two figures", "two images")
QUERY_FIGURE_ALIASES = {
    "复数残差结构": ("复数残差结构", "complex residual structure"),
    "复数残差网络": ("复数残差网络", "complex residual network", "cvresnet"),
    "信号预处理框图": ("信号预处理", "signal preprocessing", "block diagram"),
    "关键输出波形": ("输出波形", "waveform", "output waveforms"),
    "1-bit 差分解调示意": ("1-bit", "差分解调", "differential demodulation"),
}
QUERY_PHRASE_SPLIT_PATTERN = re.compile(r"[、,，]|(?:\s*(?:和|以及|及|与|and)\s*)", flags=re.IGNORECASE)
QUERY_DOC_TITLE_PATTERN = re.compile(r"《[^》]+》")
QUERY_QUOTED_PATTERN = re.compile(r"[“\"']([^“”\"']{2,40})[”\"']")
QUERY_GENERIC_PHRASES = {
    "展示",
    "同时展示",
    "请展示",
    "请同时展示",
    "说明",
    "区别",
    "两者区别",
    "概括",
    "核心方法",
    "最值得一起展示",
}


def _extract_page_number(figure_key: str):
    match = re.match(r"p(\d+)_", str(figure_key or ""))
    if not match:
        return None
    return int(match.group(1))


def _extract_figure_numbers(text: str) -> list[int]:
    numbers = []
    for match in re.finditer(r"(?:figure|fig\.?|图)\s*[:：]?\s*(\d+)", str(text or ""), flags=re.IGNORECASE):
        if match.group(1).isdigit():
            number = int(match.group(1))
            if number not in numbers:
                numbers.append(number)
    return numbers


def _normalize_vector(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm <= 0:
        return arr
    return arr / norm


def _zero_vector(dimension: int) -> np.ndarray:
    return np.zeros((dimension,), dtype=np.float32)


def _caption_body(caption: str) -> str:
    clean = re.sub(r"\s+", " ", str(caption or "").strip())
    for separator in [":", "：", ".", "。"]:
        if separator in clean:
            head, tail = clean.split(separator, 1)
            if re.match(r"^(?:figure|fig\.?|图)\s*\d+\b", head.strip(), flags=re.IGNORECASE):
                body = tail.strip()
                if body:
                    return body
    return clean


def _classify_semantic_role(caption: str) -> str:
    haystack = _semantic_haystack(caption)
    if not haystack:
        return "other"

    scores = _semantic_role_scores(haystack)
    ranked_roles = sorted(
        scores.items(),
        key=lambda item: (item[1], _semantic_role_priority(item[0])),
        reverse=True,
    )
    best_role, best_score = ranked_roles[0]
    return best_role if best_score > 0 else "other"


def _classify_visual_type(caption: str) -> str:
    text = str(caption or "").strip().lower()
    body = _caption_body(caption).lower()
    haystack = f"{text} {body}".strip()
    if not haystack:
        return "unknown"
    if any(keyword in haystack for keyword in ["button", "icon", "check for updates", "screenshot", "ui", "dialog"]):
        return "ui_screenshot"
    if any(keyword in haystack for keyword in ["code snippet", "代码片段", "frame in an 802.11 packet", "packet analysis", "terminal", "console"]):
        return "code_or_terminal"
    if any(keyword in haystack for keyword in ["figure", "fig.", "图", "示意", "framework", "architecture", "pipeline", "diagram", "network", "waveform", "curve", "matrix"]):
        return "normal_figure"
    return "other"


def _dedupe_same_role_candidates(candidates: list[dict], gap: float = SAME_ROLE_DEDUPE_GAP) -> list[dict]:
    kept = []
    seen_signatures = defaultdict(list)
    for candidate in candidates:
        doc_name = str(candidate.get("doc_name") or "").strip()
        role = str(candidate.get("semantic_role") or "other").strip()
        body = re.sub(r"\s+", " ", str(candidate.get("caption_body") or "").strip().lower())
        signature = (doc_name, role, body)
        similar_items = seen_signatures[signature]
        score = float(candidate.get("fusion_score") or candidate.get("score") or 0.0)
        if similar_items:
            continue
        if any(abs(score - float(item.get("fusion_score") or item.get("score") or 0.0)) <= gap for item in similar_items):
            continue
        similar_items.append(candidate)
        kept.append(candidate)
    return kept


def _sort_candidate_list(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("fusion_score") or item.get("score") or 0.0),
            float(item.get("caption_score") or 0.0),
            float(item.get("image_score") or 0.0),
        ),
        reverse=True,
    )


def _assign_global_rank(candidates: list[dict]) -> list[dict]:
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _semantic_haystack(text: str) -> str:
    raw = str(text or "").strip().lower()
    body = _caption_body(text).lower()
    return f"{raw} {body}".strip()


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    lowered = str(text or "").lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _semantic_role_priority(role: str) -> int:
    order = {
        "comparison_result": 5,
        "performance_curve": 4,
        "method_overview": 3,
        "module_structure": 2,
        "waveform_or_signal": 1,
        "other": 0,
    }
    return order.get(str(role or "other"), 0)


def _semantic_role_scores(text: str) -> dict[str, float]:
    lowered = str(text or "").lower()
    scores = {
        "method_overview": 0.0,
        "module_structure": 0.0,
        "waveform_or_signal": 0.0,
        "performance_curve": 0.0,
        "comparison_result": 0.0,
    }
    if not lowered:
        return scores

    scores["method_overview"] += 1.4 * _count_keyword_hits(lowered, METHOD_OVERVIEW_KEYWORDS)
    scores["module_structure"] += 1.3 * _count_keyword_hits(lowered, MODULE_STRUCTURE_KEYWORDS)
    scores["waveform_or_signal"] += 1.3 * _count_keyword_hits(lowered, WAVEFORM_SIGNAL_KEYWORDS)
    scores["performance_curve"] += 1.3 * _count_keyword_hits(lowered, PERFORMANCE_OVERVIEW_KEYWORDS)
    scores["comparison_result"] += 1.2 * _count_keyword_hits(lowered, COMPARE_KEYWORDS)

    if any(token in lowered for token in ["block diagram", "框图", "流程图", "结构图", "framework", "architecture", "pipeline", "scheme"]):
        scores["method_overview"] += 2.0
    if any(token in lowered for token in ["overall", "overview", "总体", "整体", "总览", "全过程"]):
        scores["method_overview"] += 1.0

    if any(
        token in lowered
        for token in [
            "feature extraction network",
            "residual structure",
            "residual network",
            "inception module",
            "squeeze and excitation",
            "特征提取网络",
            "残差结构",
            "残差网络",
            "模块",
        ]
    ):
        scores["module_structure"] += 2.0
    elif any(token in lowered for token in ["network", "module", "卷积", "网络", "结构"]):
        scores["module_structure"] += 1.0

    if any(token in lowered for token in ["waveform", "output waveforms", "波形", "有效段", "received signal", "delay estimation"]):
        scores["waveform_or_signal"] += 2.0
    elif any(token in lowered for token in ["signal", "demodulation", "解调", "信号"]):
        scores["waveform_or_signal"] += 0.8

    if any(
        token in lowered
        for token in [
            "identification results",
            "results under different",
            "performance comparison",
            "accuracy",
            "confusion matrix",
            "error matrix",
            "识别结果",
            "识别性能",
            "准确率",
            "混淆矩阵",
        ]
    ):
        scores["performance_curve"] += 2.0
    elif any(token in lowered for token in ["results", "result", "performance", "性能", "结果"]):
        scores["performance_curve"] += 1.0

    if scores["performance_curve"] > 0 and any(token in lowered for token in ["different", "vs", "versus", "对比", "比较", "不同"]):
        scores["comparison_result"] += 1.4

    if any(token in lowered for token in ["block diagram", "框图", "流程图"]):
        scores["waveform_or_signal"] = max(0.0, scores["waveform_or_signal"] - 1.0)
    if any(token in lowered for token in ["result", "results", "性能", "准确率", "混淆矩阵"]):
        scores["module_structure"] = max(0.0, scores["module_structure"] - 0.6)

    return scores


def _extract_query_focus_phrases(query: str) -> list[str]:
    content = QUERY_DOC_TITLE_PATTERN.sub(" ", str(query or ""))
    phrases = []
    seen = set()

    def add_phrase(raw: str, from_quote: bool = False):
        phrase = str(raw or "").replace("“", " ").replace("”", " ").replace("\"", " ").replace("'", " ")
        phrase = re.sub(r"\s+", " ", phrase).strip(" \t\n\r：:，,。.;；!?！？")
        for _ in range(3):
            trimmed = re.sub(
                r"^(?:中的|中|里|关于|有关|请同时展示|请展示|同时展示|展示|说明|并说明|如果要|哪两张图最值得一起展示|最值得一起展示)",
                "",
                phrase,
            ).strip()
            if trimmed == phrase:
                break
            phrase = trimmed
        phrase = re.sub(r"(?:并说明.*|并解释.*|有什么区别.*|两者区别.*)$", "", phrase).strip()
        phrase = re.sub(r"(?:\s+(?:图|figure|fig\.?|image|images))$", "", phrase, flags=re.IGNORECASE).strip()
        if len(phrase) < 2 or len(phrase) > 32:
            return
        if phrase.lower() in QUERY_GENERIC_PHRASES:
            return
        if not from_quote and not (
            any(token in phrase.lower() for token in ["图", "框", "结构", "网络", "模块", "波形", "解调", "diagram", "framework", "network", "module", "waveform", "demodulation"])
            or any(char.isdigit() for char in phrase)
        ):
            return
        lowered = phrase.lower()
        if lowered not in seen:
            seen.add(lowered)
            phrases.append(phrase)

    for match in QUERY_QUOTED_PATTERN.finditer(content):
        add_phrase(match.group(1), from_quote=True)

    for part in QUERY_PHRASE_SPLIT_PATTERN.split(content):
        add_phrase(part)

    return phrases[:4]


def _count_alias_hits(query: str) -> int:
    lowered_query = str(query or "").lower()
    matched = set()
    for aliases in QUERY_FIGURE_ALIASES.values():
        if any(alias.lower() in lowered_query for alias in aliases):
            matched.add(tuple(alias.lower() for alias in aliases))
    return max(len(matched), len(_extract_query_focus_phrases(query)))


def _query_role_preferences(query: str) -> dict[str, float]:
    lowered = str(query or "").lower()
    scores = _semantic_role_scores(lowered)
    if _contains_any_keyword(lowered, CORE_METHOD_KEYWORDS):
        scores["method_overview"] += 2.4
        scores["module_structure"] += 1.8
        scores["performance_curve"] = max(0.0, scores["performance_curve"] - 1.0)
        scores["comparison_result"] = max(0.0, scores["comparison_result"] - 0.6)
    if _contains_any_keyword(lowered, PAIR_REQUEST_MARKERS):
        scores["module_structure"] += 0.8
    return scores


def _infer_query_intent(query: str) -> str:
    figure_numbers = _extract_figure_numbers(query)
    focus_phrases = _extract_query_focus_phrases(query)
    alias_hits = _count_alias_hits(query)
    role_preferences = _query_role_preferences(query)
    has_visual_intent = bool(
        figure_numbers
        or focus_phrases
        or _contains_any_keyword(query, QUERY_FIGURE_ALIASES.get("复数残差结构", ()))
        or _contains_any_keyword(query, METHOD_OVERVIEW_KEYWORDS + PERFORMANCE_OVERVIEW_KEYWORDS + COMPARE_KEYWORDS)
    )
    if alias_hits >= 3 and _contains_any_keyword(query, MULTI_IMAGE_MARKERS):
        return "broad_visual"
    if alias_hits >= 2 and (_contains_any_keyword(query, COMPARE_KEYWORDS) or _contains_any_keyword(query, MULTI_IMAGE_MARKERS)):
        return "compare"
    if _contains_any_keyword(query, PAIR_REQUEST_MARKERS):
        return "compare"
    if not has_visual_intent:
        return "broad_visual"
    if _contains_any_keyword(query, METHOD_OVERVIEW_KEYWORDS):
        return "method_overview"
    if _contains_any_keyword(query, PERFORMANCE_OVERVIEW_KEYWORDS):
        return "performance_overview"
    if role_preferences["method_overview"] > max(role_preferences["performance_curve"], role_preferences["waveform_or_signal"]) + 0.8:
        return "method_overview"
    if role_preferences["performance_curve"] > role_preferences["method_overview"] + 0.8:
        return "performance_overview"
    return "broad_visual"


def _query_match_bonus(query: str, caption: str, semantic_role: str) -> float:
    lowered_query = str(query or "").lower()
    haystack = _semantic_haystack(caption)
    bonus = 0.0
    role_preferences = _query_role_preferences(query)

    focus_phrases = _extract_query_focus_phrases(query)
    phrase_hits = sum(1 for phrase in focus_phrases if phrase.lower() in haystack)
    bonus += min(0.24, 0.08 * phrase_hits)

    matched_aliases = sum(
        1
        for aliases in QUERY_FIGURE_ALIASES.values()
        if any(alias.lower() in lowered_query for alias in aliases)
        and any(alias.lower() in haystack for alias in aliases)
    )
    bonus += 0.08 * matched_aliases

    intent = _infer_query_intent(query)
    aligned_role_pref = float(role_preferences.get(semantic_role, 0.0))
    has_content_evidence = bool(phrase_hits or matched_aliases)
    if has_content_evidence:
        if aligned_role_pref >= 2.0:
            bonus += 0.10
        elif aligned_role_pref >= 0.8:
            bonus += 0.05

    if has_content_evidence and _contains_any_keyword(lowered_query, CORE_METHOD_KEYWORDS):
        if semantic_role in {"method_overview", "module_structure"}:
            bonus += 0.10
        elif semantic_role in {"performance_curve", "comparison_result"}:
            bonus -= 0.12
    if has_content_evidence and _contains_any_keyword(lowered_query, PAIR_REQUEST_MARKERS) and semantic_role == "module_structure":
        bonus += 0.05

    if intent == "method_overview":
        if semantic_role in {"method_overview", "module_structure"}:
            bonus += 0.08
        elif semantic_role in {"performance_curve", "comparison_result"}:
            bonus -= 0.06
    elif intent == "performance_overview":
        if semantic_role in {"performance_curve", "comparison_result"}:
            bonus += 0.08
        elif semantic_role in {"method_overview", "module_structure"}:
            bonus -= 0.06
    elif intent == "compare" and matched_aliases:
        bonus += 0.05
    elif intent == "broad_visual" and (matched_aliases or phrase_hits):
        bonus += 0.03
    return bonus


def _load_asset_manifest(doc_name: str, output_root=None) -> dict:
    doc_dir = get_doc_asset_dir(doc_name, output_root)
    captions = load_captions_dict(doc_name, output_root)
    asset_manifest = load_json_file(doc_dir / "manifest.json")
    if asset_manifest:
        return asset_manifest
    return {
        "run_id": derive_run_id(output_root),
        "source_file_hash": "",
        "caption_hash": captions_hash(captions),
        "figure_count": len(captions),
        "output_root": str(doc_dir.parent),
    }


def _is_index_current(index_manifest: dict, asset_manifest: dict, client: Qwen3VLEmbeddingClient, vectors_path: Path) -> bool:
    if not index_manifest or not vectors_path.is_file():
        return False
    if index_manifest.get("index_version") != IMAGE_INDEX_VERSION:
        return False
    if index_manifest.get("embedding_model") != client.model_name:
        return False
    if int(index_manifest.get("vector_dim") or 0) != client.dimension:
        return False
    if index_manifest.get("run_id") != asset_manifest.get("run_id", DEFAULT_RUN_ID):
        return False
    if index_manifest.get("source_file_hash") != asset_manifest.get("source_file_hash", ""):
        return False
    if index_manifest.get("caption_hash") != asset_manifest.get("caption_hash", ""):
        return False
    if int(index_manifest.get("figure_count") or 0) != int(asset_manifest.get("figure_count") or 0):
        return False
    try:
        vectors = np.load(vectors_path)
        if "image_vecs" not in vectors or "caption_vecs" not in vectors:
            return False
    except Exception:
        return False
    return True


def _build_failed_manifest(
    doc_name: str,
    asset_manifest: dict,
    client: Qwen3VLEmbeddingClient,
    error: str,
):
    return {
        "doc_name": doc_name,
        "run_id": asset_manifest.get("run_id", DEFAULT_RUN_ID),
        "source_file_hash": asset_manifest.get("source_file_hash", ""),
        "caption_hash": asset_manifest.get("caption_hash", ""),
        "figure_count": asset_manifest.get("figure_count", 0),
        "embedding_model": client.model_name,
        "vector_dim": client.dimension,
        "index_version": IMAGE_INDEX_VERSION,
        "status": "failed",
        "error": str(error or ""),
        "output_root": asset_manifest.get("output_root"),
        "entries": [],
    }


def _build_empty_manifest(
    doc_name: str,
    asset_manifest: dict,
    client: Qwen3VLEmbeddingClient,
    status: str,
):
    return {
        "doc_name": doc_name,
        "run_id": asset_manifest.get("run_id", DEFAULT_RUN_ID),
        "source_file_hash": asset_manifest.get("source_file_hash", ""),
        "caption_hash": asset_manifest.get("caption_hash", ""),
        "figure_count": asset_manifest.get("figure_count", 0),
        "embedding_model": client.model_name,
        "vector_dim": client.dimension,
        "index_version": IMAGE_INDEX_VERSION,
        "status": status,
        "output_root": asset_manifest.get("output_root"),
        "entries": [],
    }


def build_doc_image_index(
    doc_name: str,
    output_root=None,
    client: Qwen3VLEmbeddingClient | None = None,
    force_rebuild: bool = False,
):
    client = client or Qwen3VLEmbeddingClient()
    if not client.is_configured():
        raise ValueError("Qwen3-VL image embedding is not configured.")

    doc_dir = get_doc_asset_dir(doc_name, output_root)
    figures_dir = get_doc_figures_dir(doc_name, output_root)
    manifest_path = get_doc_image_index_manifest_path(doc_name, output_root)
    vectors_path = get_doc_image_index_vectors_path(doc_name, output_root)
    captions = load_captions_dict(doc_name, output_root)
    asset_manifest = _load_asset_manifest(doc_name, output_root)

    if not captions:
        empty_manifest = _build_empty_manifest(doc_name, asset_manifest, client, "no_figures")
        write_json_file(manifest_path, empty_manifest)
        return empty_manifest

    current_manifest = load_json_file(manifest_path)
    if not force_rebuild and _is_index_current(current_manifest, asset_manifest, client, vectors_path):
        return current_manifest

    entries = []
    image_vectors = []
    caption_vectors = []
    for figure_key, caption in captions.items():
        image_path = figures_dir / f"{figure_key}.jpg"
        if not figure_key or not image_path.is_file():
            continue
        clean_caption = str(caption or "").strip()
        image_vector = _normalize_vector(client.embed_image(image_path))
        if clean_caption:
            caption_vector = _normalize_vector(client.embed_caption(clean_caption))
        else:
            caption_vector = _zero_vector(client.dimension)
        image_vectors.append(image_vector)
        caption_vectors.append(caption_vector)
        entries.append(
            {
                "doc_name": doc_name,
                "figure_key": figure_key,
                "caption": clean_caption,
                "page": _extract_page_number(figure_key),
                "asset_relpath": build_asset_relpath(doc_name, figure_key),
                "source_file_hash": asset_manifest.get("source_file_hash", ""),
                "run_id": asset_manifest.get("run_id", DEFAULT_RUN_ID),
            }
        )

    if not entries:
        empty_manifest = _build_empty_manifest(doc_name, asset_manifest, client, "no_indexable_figures")
        write_json_file(manifest_path, empty_manifest)
        return empty_manifest

    image_vectors_array = np.asarray(image_vectors, dtype=np.float32)
    caption_vectors_array = np.asarray(caption_vectors, dtype=np.float32)
    doc_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        vectors_path,
        image_vecs=image_vectors_array,
        caption_vecs=caption_vectors_array,
    )
    index_manifest = {
        "doc_name": doc_name,
        "run_id": asset_manifest.get("run_id", DEFAULT_RUN_ID),
        "source_file_hash": asset_manifest.get("source_file_hash", ""),
        "caption_hash": asset_manifest.get("caption_hash", ""),
        "figure_count": asset_manifest.get("figure_count", len(entries)),
        "embedding_model": client.model_name,
        "vector_dim": client.dimension,
        "index_version": IMAGE_INDEX_VERSION,
        "status": "ready",
        "output_root": asset_manifest.get("output_root"),
        "entries": entries,
    }
    write_json_file(manifest_path, index_manifest)
    return index_manifest


def load_doc_image_index(doc_name: str, output_root=None):
    manifest_path = get_doc_image_index_manifest_path(doc_name, output_root)
    vectors_path = get_doc_image_index_vectors_path(doc_name, output_root)
    manifest = load_json_file(manifest_path)
    if not manifest or not vectors_path.is_file():
        empty = np.asarray([], dtype=np.float32)
        return {}, empty, empty
    data = np.load(vectors_path)
    image_vecs = data["image_vecs"] if "image_vecs" in data else np.asarray([], dtype=np.float32)
    caption_vecs = data["caption_vecs"] if "caption_vecs" in data else np.asarray([], dtype=np.float32)
    return manifest, image_vecs, caption_vecs


def ensure_doc_image_index(
    doc_name: str,
    output_root=None,
    client: Qwen3VLEmbeddingClient | None = None,
):
    client = client or Qwen3VLEmbeddingClient()
    asset_manifest = _load_asset_manifest(doc_name, output_root)
    index_manifest = load_json_file(get_doc_image_index_manifest_path(doc_name, output_root))
    vectors_path = get_doc_image_index_vectors_path(doc_name, output_root)
    if _is_index_current(index_manifest, asset_manifest, client, vectors_path):
        return index_manifest
    return build_doc_image_index(doc_name, output_root=output_root, client=client, force_rebuild=True)


def build_doc_image_index_safe(
    doc_name: str,
    output_root=None,
    client: Qwen3VLEmbeddingClient | None = None,
    force_rebuild: bool = False,
):
    client = client or Qwen3VLEmbeddingClient()
    manifest_path = get_doc_image_index_manifest_path(doc_name, output_root)
    asset_manifest = _load_asset_manifest(doc_name, output_root)
    try:
        return build_doc_image_index(
            doc_name,
            output_root=output_root,
            client=client,
            force_rebuild=force_rebuild,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build multimodal image index for document %s", doc_name)
        failed_manifest = _build_failed_manifest(doc_name, asset_manifest, client, str(exc))
        write_json_file(manifest_path, failed_manifest)
        return failed_manifest


def backfill_doc_image_indexes(
    doc_names: list[str],
    output_root=None,
    client: Qwen3VLEmbeddingClient | None = None,
    force_rebuild: bool = False,
):
    client = client or Qwen3VLEmbeddingClient()
    results = []
    for doc_name in doc_names:
        normalized_doc = str(doc_name or "").strip()
        if not normalized_doc:
            continue
        results.append(
            build_doc_image_index_safe(
                normalized_doc,
                output_root=output_root,
                client=client,
                force_rebuild=force_rebuild,
            )
        )
    return results


def get_doc_image_index_status(doc_name: str, output_root=None) -> dict:
    manifest = load_json_file(get_doc_image_index_manifest_path(doc_name, output_root))
    if not manifest:
        return {"status": "pending"}
    return {
        "status": manifest.get("status") or "pending",
        "error": manifest.get("error", ""),
        "index_version": manifest.get("index_version"),
        "figure_count": manifest.get("figure_count", 0),
    }


def retrieve_image_candidates(
    query: str,
    doc_names: list[str],
    output_root=None,
    max_candidates: int = 3,
    client: Qwen3VLEmbeddingClient | None = None,
):
    query = str(query or "").strip()
    if not query or not doc_names:
        return []

    client = client or Qwen3VLEmbeddingClient()
    if not client.is_configured():
        LOGGER.info("Skipping image retrieval because qwen3-vl-embedding is not configured.")
        return []

    try:
        query_vec = _normalize_vector(client.embed_query(query))
    except Exception:
        LOGGER.exception("Failed to embed image retrieval query.")
        return []

    seen_docs = set()
    doc_candidates = []
    max_docs = max(1, len(doc_names))
    per_doc_raw_top_k = max(PER_DOC_RAW_TOP_K, int(max_candidates or 0))
    global_candidate_top_k = max(GLOBAL_CANDIDATE_TOP_K, int(max_candidates or 0))
    for doc_rank, doc_name in enumerate(doc_names[:max_docs], start=1):
        normalized_doc = str(doc_name or "").strip()
        if not normalized_doc or normalized_doc in seen_docs:
            continue
        seen_docs.add(normalized_doc)

        try:
            manifest, image_vecs, caption_vecs = load_doc_image_index(normalized_doc, output_root=output_root)
        except Exception:
            LOGGER.exception("Failed to prepare image index for document %s", normalized_doc)
            continue

        entries = manifest.get("entries") or []
        if (
            manifest.get("status") != "ready"
            or not entries
            or image_vecs.size == 0
            or caption_vecs.size == 0
        ):
            continue

        try:
            asset_manifest = _load_asset_manifest(normalized_doc, output_root)
            vectors_path = get_doc_image_index_vectors_path(normalized_doc, output_root)
            if not _is_index_current(manifest, asset_manifest, client, vectors_path):
                LOGGER.info("Skipping stale image sidecar for document %s", normalized_doc)
                continue
        except Exception:
            LOGGER.exception("Failed to validate image sidecar for document %s", normalized_doc)
            continue

        image_sims = np.dot(image_vecs, query_vec)
        caption_sims = np.dot(caption_vecs, query_vec)
        sims = (
            IMAGE_FUSION_IMAGE_WEIGHT * image_sims
            + IMAGE_FUSION_CAPTION_WEIGHT * caption_sims
        )
        per_doc_candidates = []
        for idx, entry in enumerate(entries):
            caption = entry.get("caption", "")
            caption_body = _caption_body(caption)
            semantic_role = _classify_semantic_role(caption)
            visual_type = _classify_visual_type(caption)
            image_score = float(image_sims[idx])
            caption_score = float(caption_sims[idx])
            fusion_score = float(sims[idx]) + _query_match_bonus(query, caption, semantic_role)
            per_doc_candidates.append(
                {
                    "doc_name": entry.get("doc_name") or normalized_doc,
                    "figure_key": entry.get("figure_key"),
                    "caption": caption,
                    "page": entry.get("page"),
                    "score": fusion_score,
                    "image_score": image_score,
                    "caption_score": caption_score,
                    "fusion_score": fusion_score,
                    "caption_body": caption_body,
                    "semantic_role": semantic_role,
                    "visual_type": visual_type,
                    "doc_rank": doc_rank,
                    "asset_relpath": entry.get("asset_relpath") or build_asset_relpath(normalized_doc, entry.get("figure_key")),
                    "source": IMAGE_INDEX_SOURCE,
                }
            )
        per_doc_candidates = _sort_candidate_list(per_doc_candidates)
        per_doc_candidates = _dedupe_same_role_candidates(per_doc_candidates, gap=SAME_ROLE_DEDUPE_GAP)
        doc_candidates.extend(per_doc_candidates[:per_doc_raw_top_k])

    doc_candidates = _sort_candidate_list(doc_candidates)
    merged_candidates = []
    role_seen_by_doc = defaultdict(set)
    overflow_candidates = []
    for candidate in doc_candidates:
        doc_name = str(candidate.get("doc_name") or "").strip()
        semantic_role = str(candidate.get("semantic_role") or "other").strip()
        if semantic_role not in role_seen_by_doc[doc_name]:
            role_seen_by_doc[doc_name].add(semantic_role)
            merged_candidates.append(candidate)
        else:
            overflow_candidates.append(candidate)

    merged_candidates.extend(overflow_candidates)
    merged_candidates = _sort_candidate_list(merged_candidates)
    merged_candidates = merged_candidates[:global_candidate_top_k]
    merged_candidates = _assign_global_rank(merged_candidates)
    return merged_candidates[: max(1, int(max_candidates or 3))]
