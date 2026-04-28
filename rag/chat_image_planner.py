import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.image_asset_utils import build_image_url, normalize_doc_base_name


LOGGER = logging.getLogger(__name__)

_FIGURE_NUMBER_PATTERN = re.compile(r"(?:figure|fig\.?|图)\s*[:：]?\s*(\d+)", flags=re.IGNORECASE)
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)")
_HTML_IMAGE_BLOCK_PATTERN = re.compile(r"<div\b[^>]*>.*?<img\b[^>]*>.*?</div>", flags=re.IGNORECASE | re.DOTALL)
_HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE)
_IMAGE_INTRO_LINE_PATTERN = re.compile(
    r"^\s*(?:如下图所示[:：]?|下图所示[:：]?|请看下图[:：]?|As shown in the figure below:?|As shown in the figures below:?|See the figure below:?|See the figures below:?)\s*$",
    flags=re.IGNORECASE,
)
_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", flags=re.DOTALL | re.IGNORECASE)
_NOT_FOUND_PATTERNS = [
    re.compile(r"知识库中未找到您要的答案", flags=re.IGNORECASE),
    re.compile(r"知识库中未找到相关内容", flags=re.IGNORECASE),
    re.compile(r"not found in the knowledge base", flags=re.IGNORECASE),
    re.compile(r"no relevant content was found in the knowledge base", flags=re.IGNORECASE),
]
_CANNOT_DISPLAY_PATTERNS = [
    re.compile(r"无法直接展示图片", flags=re.IGNORECASE),
    re.compile(r"无法展示图片", flags=re.IGNORECASE),
    re.compile(r"无法直接配图", flags=re.IGNORECASE),
    re.compile(r"无法配图", flags=re.IGNORECASE),
    re.compile(r"未提供.*图像文件", flags=re.IGNORECASE),
    re.compile(r"cannot directly display (the )?image", flags=re.IGNORECASE),
    re.compile(r"unable to display (the )?image", flags=re.IGNORECASE),
]

_RENDER_MODES = {"none", "single", "compare", "gallery"}
_PRESENTATION_BY_MODE = {
    "none": "focus",
    "single": "focus",
    "compare": "side_by_side",
    "gallery": "grid",
}
_MODE_LIMITS = {
    "none": 0,
    "single": 1,
    "compare": 2,
    "gallery": 3,
}
_MIN_CONFIDENCE = 0.35
_CANDIDATE_SUMMARY_LIMIT = 5
_IMAGE_GATE_THRESHOLDS = {
    "score_floor": 0.35,
    "single_high_conf": 0.62,
    "single_gap_min": 0.08,
    "broad_single_gap_min": 0.14,
    "compare_floor": 0.46,
    "compare_gap_max": 0.10,
    "gallery_floor": 0.42,
    "gallery_top3_mean_min": 0.46,
    "specific_figure_floor": 0.42,
    "text_only_single_floor": 0.62,
    "ui_penalty": -0.20,
    "code_penalty": -0.12,
    "weak_caption_penalty": -0.08,
}
_IMAGE_VISUAL_KEYWORDS = (
    "图",
    "图片",
    "配图",
    "示意图",
    "figure",
    "fig.",
    "illustration",
    "diagram",
    "image",
)
_COMPARE_KEYWORDS = ("compare", "comparison", "对比", "比较", "区别", "difference", "versus", "vs")
_METHOD_OVERVIEW_KEYWORDS = (
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
    "方案",
    "示意图",
    "block diagram",
    "architecture",
    "framework",
    "pipeline",
    "overview",
    "method overview",
    "flow chart",
    "scheme",
    "based identification",
    "identification framework",
)
_PERFORMANCE_OVERVIEW_KEYWORDS = (
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
_MODULE_STRUCTURE_KEYWORDS = (
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
)
_WAVEFORM_SIGNAL_KEYWORDS = (
    "waveform",
    "waveforms",
    "received signal",
    "valid segment",
    "delay estimation",
    "demodulation",
    "signal waveform",
    "波形",
    "输出波形",
    "接收信号",
    "有效段",
    "延时估计",
    "解调",
)
_CORE_METHOD_KEYWORDS = (
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
_PAIR_REQUEST_MARKERS = ("两张图", "2张图", "2 张图", "two figures", "two images")
_MULTI_IMAGE_MARKERS = ("同时展示", "分别展示", "多张图", "几张图")
_QUERY_FIGURE_ALIASES = {
    "复数残差结构": ("复数残差结构", "complex residual structure"),
    "复数残差网络": ("复数残差网络", "complex residual network", "cvresnet"),
    "信号预处理框图": ("信号预处理", "signal preprocessing", "block diagram"),
    "关键输出波形": ("输出波形", "waveform", "output waveforms"),
    "1-bit 差分解调示意": ("1-bit", "差分解调", "differential demodulation"),
}
_QUERY_PHRASE_SPLIT_PATTERN = re.compile(r"[、,，]|(?:\s*(?:和|以及|及|与|and)\s*)", flags=re.IGNORECASE)
_QUERY_DOC_TITLE_PATTERN = re.compile(r"《[^》]+》")
_QUERY_QUOTED_PATTERN = re.compile(r"[“\"']([^“”\"']{2,40})[”\"']")
_QUERY_GENERIC_PHRASES = {
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


def resolve_image_server_base_url() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    return (os.getenv("SERVER_IP") or "http://127.0.0.1:8000").strip()


def _extract_figure_numbers(text: str) -> list[int]:
    numbers = []
    for match in _FIGURE_NUMBER_PATTERN.finditer(str(text or "")):
        if match.group(1).isdigit():
            number = int(match.group(1))
            if number not in numbers:
                numbers.append(number)
    return numbers


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = _JSON_BLOCK_PATTERN.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            LOGGER.warning("Failed to parse image render plan JSON: %s", raw)
            return {}
    return data if isinstance(data, dict) else {}


def strip_legacy_answer_images(answer: str) -> str:
    text = str(answer or "")
    text = _HTML_IMAGE_BLOCK_PATTERN.sub("", text)
    text = _HTML_IMAGE_PATTERN.sub("", text)
    text = _MARKDOWN_IMAGE_PATTERN.sub("", text)
    cleaned_lines = []
    for line in text.splitlines():
        if _IMAGE_INTRO_LINE_PATTERN.match(line.strip()):
            continue
        cleaned_lines.append(line.rstrip())
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_evidence_summary(kbinfos: dict, max_chunks: int = 3, max_chars: int = 1400) -> str:
    parts = []
    for chunk in (kbinfos or {}).get("chunks", [])[:max_chunks]:
        preview = str(chunk.get("content_with_weight") or chunk.get("content_ltks") or "").strip()
        if preview:
            parts.append(preview[:600])
    summary = "\n\n".join(parts)
    return summary[:max_chars]


def _build_default_plan(reason_code: str = "no_relevant_image") -> dict[str, Any]:
    return {
        "show_images": False,
        "mode": "none",
        "presentation": _PRESENTATION_BY_MODE["none"],
        "items": [],
        "reason_code": reason_code,
        "confidence": 0.0,
        "planner_source": "planner",
    }


def _build_display_caption(caption: str, figure_key: str) -> str:
    clean_caption = re.sub(r"\s+", " ", str(caption or "").strip())
    if clean_caption:
        if re.match(r"^(?:figure|fig\.?|图)\s*\d+\b", clean_caption, flags=re.IGNORECASE):
            return clean_caption
        numbers = _extract_figure_numbers(clean_caption)
        if numbers:
            return f"Figure {numbers[0]}. {clean_caption}"
        page_match = re.match(r"p(\d+)_", str(figure_key or ""))
        if page_match:
            return f"Figure {page_match.group(1)}. {clean_caption}"
        return clean_caption
    page_match = re.match(r"p(\d+)_", str(figure_key or ""))
    if page_match:
        return f"Figure {page_match.group(1)}"
    return "Relevant figure"


def _score_bucket(score: float) -> str:
    if score >= 0.8:
        return "very_high"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    if score >= 0.2:
        return "low"
    return "very_low"


def _caption_visual_type(caption: str) -> str:
    text = str(caption or "").strip().lower()
    if not text:
        return "unknown"
    if any(keyword in text for keyword in ["button", "icon", "check for updates", "screenshot", "ui", "dialog"]):
        return "ui_screenshot"
    if any(keyword in text for keyword in ["code snippet", "代码片段", "frame in an 802.11 packet", "packet analysis"]):
        return "code_or_terminal"
    if any(keyword in text for keyword in ["figure", "fig.", "图 ", "图：", "示意", "framework", "architecture", "pipeline", "waveform", "curve", "matrix", "block diagram"]):
        return "figure_or_diagram"
    return "other"


def _gate_visual_type(caption: str) -> str:
    visual_type = _caption_visual_type(caption)
    body = _caption_body(caption)
    if re.match(r"^(?:figure|fig\.?|图)\s*\d+\b", str(caption or "").strip(), flags=re.IGNORECASE):
        return "normal_figure"
    if visual_type == "ui_screenshot":
        return "ui_screenshot"
    if visual_type == "code_or_terminal":
        return "code_or_terminal"
    if len(str(body or "").strip()) < 12:
        return "weak_caption"
    if visual_type == "figure_or_diagram":
        return "normal_figure"
    return "other"


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in keywords)


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

    scores["method_overview"] += 1.4 * _count_keyword_hits(lowered, _METHOD_OVERVIEW_KEYWORDS)
    scores["module_structure"] += 1.3 * _count_keyword_hits(lowered, _MODULE_STRUCTURE_KEYWORDS)
    scores["waveform_or_signal"] += 1.3 * _count_keyword_hits(lowered, _WAVEFORM_SIGNAL_KEYWORDS)
    scores["performance_curve"] += 1.3 * _count_keyword_hits(lowered, _PERFORMANCE_OVERVIEW_KEYWORDS)
    scores["comparison_result"] += 1.2 * _count_keyword_hits(lowered, _COMPARE_KEYWORDS)

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
    content = _QUERY_DOC_TITLE_PATTERN.sub(" ", str(query or ""))
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
        if phrase.lower() in _QUERY_GENERIC_PHRASES:
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

    for match in _QUERY_QUOTED_PATTERN.finditer(content):
        add_phrase(match.group(1), from_quote=True)

    for part in _QUERY_PHRASE_SPLIT_PATTERN.split(content):
        add_phrase(part)

    return phrases[:4]


def _count_alias_hits(query: str) -> int:
    lowered_query = str(query or "").lower()
    matched = set()
    for aliases in _QUERY_FIGURE_ALIASES.values():
        if any(alias.lower() in lowered_query for alias in aliases):
            matched.add(tuple(alias.lower() for alias in aliases))
    return max(len(matched), len(_extract_query_focus_phrases(query)))


def _query_role_preferences(query: str) -> dict[str, float]:
    lowered = str(query or "").lower()
    scores = _semantic_role_scores(lowered)
    if _contains_any_keyword(lowered, _CORE_METHOD_KEYWORDS):
        scores["method_overview"] += 2.4
        scores["module_structure"] += 1.8
        scores["performance_curve"] = max(0.0, scores["performance_curve"] - 1.0)
        scores["comparison_result"] = max(0.0, scores["comparison_result"] - 0.6)
    if _contains_any_keyword(lowered, _PAIR_REQUEST_MARKERS):
        scores["module_structure"] += 0.8
    return scores


def _infer_query_type(query: str) -> str:
    figure_numbers = _extract_figure_numbers(query)
    focus_phrases = _extract_query_focus_phrases(query)
    alias_hits = _count_alias_hits(query)
    role_preferences = _query_role_preferences(query)
    has_visual_intent = bool(
        figure_numbers
        or focus_phrases
        or _contains_any_keyword(query, _IMAGE_VISUAL_KEYWORDS)
        or _contains_any_keyword(query, _MULTI_IMAGE_MARKERS)
        or _contains_any_keyword(query, _PAIR_REQUEST_MARKERS)
    )
    if alias_hits >= 3 and _contains_any_keyword(query, _MULTI_IMAGE_MARKERS):
        return "broad_visual"
    if alias_hits >= 2:
        return "compare"
    if _contains_any_keyword(query, _PAIR_REQUEST_MARKERS):
        return "compare"
    if len(figure_numbers) >= 2 or _contains_any_keyword(query, _COMPARE_KEYWORDS):
        return "compare"
    if figure_numbers:
        return "specific_figure"
    if not has_visual_intent:
        return "text_only"
    if _contains_any_keyword(query, _METHOD_OVERVIEW_KEYWORDS):
        return "method_overview"
    if _contains_any_keyword(query, _PERFORMANCE_OVERVIEW_KEYWORDS):
        return "performance_overview"
    if role_preferences["method_overview"] > max(role_preferences["performance_curve"], role_preferences["waveform_or_signal"]) + 0.8:
        return "method_overview"
    if role_preferences["performance_curve"] > role_preferences["method_overview"] + 0.8:
        return "performance_overview"
    if _contains_any_keyword(query, _IMAGE_VISUAL_KEYWORDS):
        return "broad_visual"
    return "text_only"


def _infer_semantic_role(candidate: dict[str, Any]) -> str:
    role = str(candidate.get("semantic_role") or "").strip()
    if role:
        return role
    caption = str(candidate.get("caption") or "").strip().lower()
    body = _caption_body(caption).lower()
    haystack = f"{caption} {body}".strip()
    scores = _semantic_role_scores(haystack)
    ranked_roles = sorted(
        scores.items(),
        key=lambda item: (item[1], _semantic_role_priority(item[0])),
        reverse=True,
    )
    best_role, best_score = ranked_roles[0]
    return best_role if best_score > 0 else "other"


def _role_alignment(query_type: str, semantic_role: str) -> str:
    if query_type == "method_overview":
        return "aligned" if semantic_role in {"method_overview", "module_structure"} else "misaligned"
    if query_type == "performance_overview":
        return "aligned" if semantic_role in {"performance_curve", "comparison_result"} else "misaligned"
    if query_type == "compare":
        return "aligned" if semantic_role in {"method_overview", "module_structure", "waveform_or_signal", "comparison_result"} else "neutral"
    return "neutral"


def _candidate_text_matches_query(candidate: dict[str, Any], query: str) -> bool:
    haystack = f"{candidate.get('caption', '')} {candidate.get('caption_body', '')}".lower()
    lowered_query = str(query or "").lower()
    for phrase in _extract_query_focus_phrases(query):
        if phrase.lower() in haystack:
            return True
    for aliases in _QUERY_FIGURE_ALIASES.values():
        matched_query = any(alias.lower() in lowered_query for alias in aliases)
        matched_candidate = any(alias.lower() in haystack for alias in aliases)
        if matched_query and matched_candidate:
            return True
    return False


def _candidate_query_utility(candidate: dict[str, Any], query: str) -> float:
    role = _infer_semantic_role(candidate)
    preferences = _query_role_preferences(query)
    phrase_hits = sum(
        1 for phrase in _extract_query_focus_phrases(query)
        if phrase.lower() in f"{candidate.get('caption', '')} {candidate.get('caption_body', '')}".lower()
    )
    utility = float(candidate.get("gate_adjusted_score") or candidate.get("score") or 0.0)
    utility += min(0.24, 0.08 * phrase_hits)
    utility += min(0.14, 0.05 * max(0.0, float(preferences.get(role, 0.0))))
    if preferences["method_overview"] > preferences["performance_curve"] + 0.8 and role in {"performance_curve", "comparison_result"}:
        utility -= 0.10
    if preferences["performance_curve"] > preferences["method_overview"] + 0.8 and role in {"method_overview", "module_structure"}:
        utility -= 0.08
    if str(candidate.get("gate_visual_type") or "other") == "other":
        utility -= 0.03
    return utility


def _prefer_primary_doc_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    doc_ranks = [int(item.get("doc_rank") or 0) for item in candidates if int(item.get("doc_rank") or 0) > 0]
    if not doc_ranks:
        return candidates
    primary_rank = min(doc_ranks)
    primary = [item for item in candidates if int(item.get("doc_rank") or 0) == primary_rank]
    return primary if len(primary) >= 2 else candidates


def _select_compare_candidates(scored_candidates: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    lowered_query = str(query or "").lower()
    focus_phrases = _extract_query_focus_phrases(query)
    selected = []
    used = set()
    for phrase in focus_phrases:
        match = next(
            iter(
                sorted(
                    [
                        item
                        for item in scored_candidates
                        if (item.get("doc_name"), item.get("figure_key")) not in used
                        and phrase.lower() in f"{item.get('caption', '')} {item.get('caption_body', '')}".lower()
                    ],
                    key=lambda item: _candidate_query_utility(item, query),
                    reverse=True,
                )
            ),
            None,
        )
        if match:
            used.add((match.get("doc_name"), match.get("figure_key")))
            selected.append(match)
        if len(selected) >= 2:
            return selected[:2]

    matched_alias_groups = [
        aliases
        for aliases in _QUERY_FIGURE_ALIASES.values()
        if any(alias.lower() in lowered_query for alias in aliases)
    ]
    for aliases in matched_alias_groups:
        def alias_match_score(item: dict[str, Any]) -> tuple[int, int, float]:
            caption = str(item.get("caption") or "").strip().lower()
            text = f"{caption} {item.get('caption_body', '')}".lower()
            exact_label = any(
                re.match(rf"^(?:figure|fig\.?|图)\s*\d+\s*{re.escape(alias.lower())}$", caption)
                or caption == alias.lower()
                or str(item.get("caption_body") or "").strip().lower() == alias.lower()
                for alias in aliases
            )
            strong = any(
                text.startswith(alias.lower())
                or alias.lower() in caption
                for alias in aliases
            )
            role_bonus = 1 if str(item.get("semantic_role") or "") in {"module_structure", "waveform_or_signal"} else 0
            short_bonus = max(0.0, 1.0 - min(len(caption), 120) / 120.0)
            return (1 if exact_label else 0, 1 if strong else 0, role_bonus + _candidate_query_utility(item, query) + short_bonus)

        match = next(
            iter(
                sorted(
                    [
                        item
                        for item in scored_candidates
                        if (item.get("doc_name"), item.get("figure_key")) not in used
                        and any(alias.lower() in f"{item.get('caption', '')} {item.get('caption_body', '')}".lower() for alias in aliases)
                    ],
                    key=alias_match_score,
                    reverse=True,
                )
            ),
            None,
        )
        if match:
            used.add((match.get("doc_name"), match.get("figure_key")))
            selected.append(match)
        if len(selected) >= 2:
            return selected[:2]

    for item in scored_candidates:
        key = (item.get("doc_name"), item.get("figure_key"))
        if key in used:
            continue
        if _query_role_preferences(query)["method_overview"] > _query_role_preferences(query)["performance_curve"] + 0.8:
            role = _infer_semantic_role(item)
            if role in {"performance_curve", "comparison_result"} and any(
                _infer_semantic_role(other) in {"method_overview", "module_structure"}
                for other in scored_candidates
                if (other.get("doc_name"), other.get("figure_key")) not in used
            ):
                continue
        selected.append(item)
        used.add(key)
        if len(selected) >= 2:
            break
    if len(selected) < 2:
        fallback = sorted(
            [item for item in scored_candidates if (item.get("doc_name"), item.get("figure_key")) not in used],
            key=lambda item: _candidate_query_utility(item, query),
            reverse=True,
        )
        selected.extend(fallback[: 2 - len(selected)])
    return selected[:2]


def _select_gallery_candidates(scored_candidates: list[dict[str, Any]], query: str, limit: int = 3) -> list[dict[str, Any]]:
    lowered_query = str(query or "").lower()
    focus_phrases = _extract_query_focus_phrases(query)
    selected = []
    used = set()
    for phrase in focus_phrases:
        match = next(
            iter(
                sorted(
                    [
                        item
                        for item in scored_candidates
                        if (item.get("doc_name"), item.get("figure_key")) not in used
                        and phrase.lower() in f"{item.get('caption', '')} {item.get('caption_body', '')}".lower()
                    ],
                    key=lambda item: _candidate_query_utility(item, query),
                    reverse=True,
                )
            ),
            None,
        )
        if match:
            key = (match.get("doc_name"), match.get("figure_key"))
            used.add(key)
            selected.append(match)
        if len(selected) >= limit:
            return selected[:limit]

    for aliases in _QUERY_FIGURE_ALIASES.values():
        if not any(alias.lower() in lowered_query for alias in aliases):
            continue
        match = next(
            iter(
                sorted(
                    [
                        item
                        for item in scored_candidates
                        if (item.get("doc_name"), item.get("figure_key")) not in used
                        and any(alias.lower() in f"{item.get('caption', '')} {item.get('caption_body', '')}".lower() for alias in aliases)
                    ],
                    key=lambda item: _candidate_query_utility(item, query),
                    reverse=True,
                )
            ),
            None,
        )
        if match:
            key = (match.get("doc_name"), match.get("figure_key"))
            used.add(key)
            selected.append(match)
        if len(selected) >= limit:
            return selected[:limit]

    for item in sorted(scored_candidates, key=lambda candidate: _candidate_query_utility(candidate, query), reverse=True):
        key = (item.get("doc_name"), item.get("figure_key"))
        if key in used:
            continue
        selected.append(item)
        used.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _text_evidence_mentions_figure(kbinfos: dict) -> bool:
    evidence_summary = _build_evidence_summary(kbinfos, max_chunks=4, max_chars=1600)
    return bool(_extract_figure_numbers(evidence_summary) or _contains_any_keyword(evidence_summary, _IMAGE_VISUAL_KEYWORDS))


def _candidate_matches_requested_figure(candidate: dict[str, Any], requested_numbers: list[int]) -> bool:
    if not requested_numbers:
        return False

    caption_numbers = _extract_figure_numbers(str(candidate.get("caption") or ""))
    if any(number in caption_numbers for number in requested_numbers):
        return True

    figure_key = str(candidate.get("figure_key") or "")
    page_match = re.match(r"p(\d+)_", figure_key)
    if page_match:
        page_number = int(page_match.group(1))
        if page_number in requested_numbers:
            return True
    return False


def _score_with_penalty(candidate: dict[str, Any]) -> tuple[float, str]:
    score = float(candidate.get("fusion_score") or candidate.get("score") or 0.0)
    visual_type = _gate_visual_type(candidate.get("caption", ""))
    adjusted_score = score
    if visual_type == "ui_screenshot":
        adjusted_score += _IMAGE_GATE_THRESHOLDS["ui_penalty"]
    elif visual_type == "code_or_terminal":
        adjusted_score += _IMAGE_GATE_THRESHOLDS["code_penalty"]
    elif visual_type == "weak_caption":
        adjusted_score += _IMAGE_GATE_THRESHOLDS["weak_caption_penalty"]
    return max(0.0, min(1.0, adjusted_score)), visual_type


def evaluate_image_gate(query: str, kbinfos: dict) -> dict[str, Any]:
    candidates = [dict(item) for item in ((kbinfos or {}).get("image_candidates") or []) if isinstance(item, dict)]
    query = str(query or "").strip()
    if not query or not candidates:
        return {
            "passed": False,
            "mode_hint": "none",
            "reason_code": "no_image_candidates",
            "failure_stage": "retrieval",
            "top1_score": None,
            "top1_top2_gap": None,
            "selected_candidate_keys": [],
            "role_alignment": "unknown",
            "filtered_candidates": [],
            "score_breakdown": {
                "query_type": _infer_query_type(query),
                "candidate_count": len(candidates),
            },
        }

    scored_candidates = []
    for rank, candidate in enumerate(candidates, start=1):
        adjusted_score, visual_type = _score_with_penalty(candidate)
        enriched = dict(candidate)
        enriched["rank"] = int(candidate.get("rank") or rank)
        enriched["gate_adjusted_score"] = adjusted_score
        enriched["gate_visual_type"] = visual_type
        enriched["semantic_role"] = _infer_semantic_role(candidate)
        enriched["role_alignment"] = _role_alignment(_infer_query_type(query), enriched["semantic_role"])
        enriched["figure_label_match"] = _candidate_matches_requested_figure(candidate, _extract_figure_numbers(query))
        scored_candidates.append(enriched)
    scored_candidates.sort(
        key=lambda item: (float(item.get("gate_adjusted_score") or 0.0), float(item.get("score") or 0.0)),
        reverse=True,
    )

    top1_score = float(scored_candidates[0].get("gate_adjusted_score") or 0.0)
    top2_score = float(scored_candidates[1].get("gate_adjusted_score") or 0.0) if len(scored_candidates) > 1 else 0.0
    top3_scores = [float(item.get("gate_adjusted_score") or 0.0) for item in scored_candidates[:3]]
    top3_mean = sum(top3_scores) / len(top3_scores)
    top1_top2_gap = top1_score - top2_score if len(scored_candidates) > 1 else top1_score
    support_count_above_floor = sum(
        1
        for item in scored_candidates
        if float(item.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["score_floor"]
    )
    requested_numbers = _extract_figure_numbers(query)
    query_type = _infer_query_type(query)
    explicit_match_candidates = [
        item for item in scored_candidates if _candidate_matches_requested_figure(item, requested_numbers)
    ]
    explicit_figure_match = bool(explicit_match_candidates)
    text_evidence_mentions_figure = _text_evidence_mentions_figure(kbinfos)
    best_visual_type = str(scored_candidates[0].get("gate_visual_type") or "other")

    debug = {
        "query_type": query_type,
        "candidate_count": len(scored_candidates),
        "top1_score": round(top1_score, 6),
        "top2_score": round(top2_score, 6),
        "top3_mean": round(top3_mean, 6),
        "top1_top2_gap": round(top1_top2_gap, 6),
        "support_count_above_floor": support_count_above_floor,
        "explicit_figure_match": explicit_figure_match,
        "text_evidence_mentions_figure": text_evidence_mentions_figure,
        "best_visual_type": best_visual_type,
    }

    def selected_keys(items: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        return [
            {"doc_name": str(item.get("doc_name") or "").strip(), "figure_key": str(item.get("figure_key") or "").strip()}
            for item in (items or [])
            if str(item.get("doc_name") or "").strip() and str(item.get("figure_key") or "").strip()
        ]

    def build_result(
        passed: bool,
        mode_hint: str,
        reason_code: str,
        selected: list[dict[str, Any]] | None = None,
        failure_stage: str | None = None,
        role_alignment: str | None = None,
    ):
        return {
            "passed": passed,
            "mode_hint": mode_hint,
            "reason_code": reason_code,
            "failure_stage": failure_stage or ("none" if passed else "gate"),
            "top1_score": round(top1_score, 6),
            "top1_top2_gap": round(top1_top2_gap, 6),
            "selected_candidate_keys": selected_keys(selected),
            "role_alignment": role_alignment or (selected[0].get("role_alignment") if selected else "unknown"),
            "filtered_candidates": [dict(item) for item in (selected or [])],
            "score_breakdown": debug,
            "gate_score_breakdown": debug,
            "query_type": query_type,
        }

    if query_type == "specific_figure":
        if not explicit_match_candidates:
            return build_result(False, "none", "specific_figure_no_match", failure_stage="retrieval", role_alignment="unknown")
        matched = explicit_match_candidates[0]
        if float(matched.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["specific_figure_floor"]:
            return build_result(True, "single", "specific_figure_match", [matched], role_alignment="aligned")
        return build_result(False, "none", "specific_figure_low_score", [matched], "gate", "aligned")

    if query_type == "compare":
        if len(scored_candidates) < 2:
            return build_result(False, "none", "compare_insufficient_candidates", failure_stage="retrieval")
        scored_candidates = _prefer_primary_doc_candidates(scored_candidates)
        compare_pool = [
            item
            for item in scored_candidates
            if str(item.get("gate_visual_type") or "other") not in {"ui_screenshot", "code_or_terminal"}
        ]
        if explicit_match_candidates:
            compare_pool = explicit_match_candidates + [item for item in compare_pool if item not in explicit_match_candidates]
        else:
            text_match_candidates = _select_compare_candidates(compare_pool, query)
            if text_match_candidates:
                compare_pool = text_match_candidates + [item for item in compare_pool if item not in text_match_candidates]
        first_two = compare_pool[:2]
        both_strong = all(
            float(item.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["compare_floor"]
            for item in first_two
        )
        same_doc = len({str(item.get("doc_name") or "").strip() for item in first_two}) == 1
        if len(first_two) == 2 and both_strong and same_doc:
            return build_result(True, "compare", "compare_two_strong_candidates", first_two, role_alignment="aligned")
        return build_result(False, "none", "compare_candidates_not_stable", first_two, "gate", "misaligned")

    if query_type == "method_overview":
        scoped_candidates = _prefer_primary_doc_candidates(scored_candidates)
        aligned = [item for item in scoped_candidates if item.get("semantic_role") in {"method_overview", "module_structure"}]
        if not aligned:
            return build_result(False, "none", "method_overview_role_mismatch", failure_stage="retrieval", role_alignment="misaligned")
        best = aligned[0]
        if float(best.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["score_floor"]:
            return build_result(True, "single", "method_overview_role_aligned", [best], role_alignment="aligned")
        return build_result(False, "none", "method_overview_low_score", [best], "gate", "aligned")

    if query_type == "performance_overview":
        scoped_candidates = _prefer_primary_doc_candidates(scored_candidates)
        aligned = [item for item in scoped_candidates if item.get("semantic_role") in {"performance_curve", "comparison_result"}]
        if not aligned:
            return build_result(False, "none", "performance_overview_role_mismatch", failure_stage="retrieval", role_alignment="misaligned")
        best = aligned[0]
        if float(best.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["score_floor"]:
            return build_result(True, "single", "performance_overview_role_aligned", [best], role_alignment="aligned")
        return build_result(False, "none", "performance_overview_low_score", [best], "gate", "aligned")

    if query_type == "broad_visual":
        scored_candidates = _prefer_primary_doc_candidates(scored_candidates)
        aligned_candidates = [item for item in scored_candidates if item.get("role_alignment") != "misaligned"]
        if aligned_candidates:
            scored_candidates = aligned_candidates
            top1_score = float(scored_candidates[0].get("gate_adjusted_score") or 0.0)
            top2_score = float(scored_candidates[1].get("gate_adjusted_score") or 0.0) if len(scored_candidates) > 1 else 0.0
            top3_scores = [float(item.get("gate_adjusted_score") or 0.0) for item in scored_candidates[:3]]
            top3_mean = sum(top3_scores) / len(top3_scores)
            top1_top2_gap = top1_score - top2_score if len(scored_candidates) > 1 else top1_score
            best_visual_type = str(scored_candidates[0].get("gate_visual_type") or "other")
        alias_gallery = _count_alias_hits(query) >= 3 and _contains_any_keyword(query, _MULTI_IMAGE_MARKERS)
        if alias_gallery:
            gallery_candidates = _select_gallery_candidates(scored_candidates, query, limit=3)
            if len(gallery_candidates) == 3 and all(
                float(item.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["gallery_floor"]
                for item in gallery_candidates
            ):
                return build_result(True, "gallery", "broad_visual_alias_gallery", gallery_candidates, role_alignment="aligned")
        if (
            top1_score >= _IMAGE_GATE_THRESHOLDS["single_high_conf"]
            and top1_top2_gap >= _IMAGE_GATE_THRESHOLDS["broad_single_gap_min"]
            and best_visual_type == "normal_figure"
        ):
            return build_result(True, "single", "broad_visual_clear_head", [scored_candidates[0]], role_alignment="aligned")

        if (
            len(scored_candidates) >= 2
            and top1_score >= 0.55
            and top2_score >= 0.50
            and top1_top2_gap <= _IMAGE_GATE_THRESHOLDS["compare_gap_max"]
            and all(
                str(item.get("gate_visual_type") or "other") == "normal_figure"
                for item in scored_candidates[:2]
            )
        ):
            return build_result(True, "compare", "broad_visual_two_strong_candidates", scored_candidates[:2], role_alignment="aligned")

        if (
            support_count_above_floor >= 3
            and top3_mean >= _IMAGE_GATE_THRESHOLDS["gallery_top3_mean_min"]
            and sum(
                1
                for item in scored_candidates[:3]
                if float(item.get("gate_adjusted_score") or 0.0) >= _IMAGE_GATE_THRESHOLDS["gallery_floor"]
            )
            >= 3
        ):
            return build_result(True, "gallery", "broad_visual_three_supported_candidates", scored_candidates[:3], role_alignment="aligned")

        return build_result(False, "none", "broad_visual_not_confident", scored_candidates[:3], "gate", "neutral")

    if (
        text_evidence_mentions_figure
        and top1_score >= _IMAGE_GATE_THRESHOLDS["text_only_single_floor"]
        and top1_top2_gap >= _IMAGE_GATE_THRESHOLDS["single_gap_min"]
        and best_visual_type == "normal_figure"
    ):
        return build_result(True, "single", "text_evidence_supports_single", [scored_candidates[0]], role_alignment="aligned")

    return build_result(False, "none", "text_only_not_visual", failure_stage="gate", role_alignment="neutral")


def _build_candidate_summary(candidates: list[dict[str, Any]], max_items: int = _CANDIDATE_SUMMARY_LIMIT) -> str:
    if not candidates:
        return "[empty]"

    top_candidates = list(candidates[: max(1, int(max_items or _CANDIDATE_SUMMARY_LIMIT))])
    max_score = max(float(item.get("score") or 0.0) for item in top_candidates)
    min_score = min(float(item.get("score") or 0.0) for item in top_candidates)
    score_span = max(max_score - min_score, 1e-6)

    lines = []
    for idx, item in enumerate(top_candidates, start=1):
        score = float(item.get("score") or 0.0)
        next_score = float(top_candidates[idx].get("score") or 0.0) if idx < len(top_candidates) else 0.0
        gap_to_next = score - next_score if idx < len(top_candidates) else score
        normalized_score = 1.0 if score_span <= 1e-6 else (score - min_score) / score_span
        lines.append(
            json.dumps(
                {
                    "rank": idx,
                    "doc_name": item.get("doc_name"),
                    "figure_key": item.get("figure_key"),
                    "score": round(score, 6),
                    "fusion_score": round(float(item.get("fusion_score") or score), 6),
                    "image_score": round(float(item.get("image_score") or 0.0), 6),
                    "caption_score": round(float(item.get("caption_score") or 0.0), 6),
                    "score_bucket": _score_bucket(score),
                    "score_norm": round(float(normalized_score), 4),
                    "gap_to_next": round(float(gap_to_next), 6),
                    "semantic_role": str(item.get("semantic_role") or _infer_semantic_role(item)),
                    "visual_type": str(item.get("visual_type") or _caption_visual_type(item.get("caption", ""))),
                    "doc_rank": int(item.get("doc_rank") or 0),
                    "figure_label_match": bool(item.get("figure_label_match")),
                    "caption": str(item.get("caption") or "")[:220],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def build_render_plan_knowledge(plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict) or not plan.get("show_images"):
        return ""

    items = [item for item in (plan.get("items") or []) if isinstance(item, dict)]
    if not items:
        return ""

    lines = [
        "Supplementary image evidence selected for the current answer:",
        "The application can render these figures directly, so do not claim that the image cannot be displayed when using this evidence.",
    ]
    for rank, item in enumerate(items, start=1):
        doc_name = str(item.get("doc_name") or "").strip()
        figure_key = str(item.get("figure_key") or "").strip()
        display_caption = str(item.get("display_caption") or item.get("caption") or "").strip()
        lines.append(
            f"- Selected figure {rank}: doc={doc_name}; figure_key={figure_key}; caption={display_caption}"
        )
    return "\n".join(lines)


def _contains_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    content = str(text or "")
    return any(pattern.search(content) for pattern in patterns)


def _filter_answer_paragraphs(answer: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", str(answer or "")) if paragraph.strip()]
    kept = []
    for paragraph in paragraphs:
        if _contains_pattern(paragraph, _CANNOT_DISPLAY_PATTERNS):
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip()


def _is_chinese_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


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


def synthesize_image_grounded_answer(query: str, plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict) or not plan.get("show_images"):
        return ""

    items = [item for item in (plan.get("items") or []) if isinstance(item, dict)]
    if not items:
        return ""

    zh = _is_chinese_text(query)
    mode = str(plan.get("mode") or "single").strip().lower()
    if mode == "compare" and len(items) >= 2:
        if zh:
            lines = ["根据检索到的图片证据，相关对比图如下："]
            for item in items[:2]:
                lines.append(f"- {item.get('display_caption') or item.get('caption')}")
            return "\n".join(lines)
        lines = ["Based on the retrieved image evidence, the relevant comparison figures are:"]
        for item in items[:2]:
            lines.append(f"- {item.get('display_caption') or item.get('caption')}")
        return "\n".join(lines)

    primary = items[0]
    caption = str(primary.get("display_caption") or primary.get("caption") or "").strip()
    body = _caption_body(caption)
    if zh:
        return f"根据检索到的图片证据，这张图主要展示的是：{body}"
    return f"According to the retrieved image evidence, this figure mainly shows: {body}"


def postprocess_answer_with_render_plan(answer: str, query: str, plan: dict[str, Any]) -> str:
    text = strip_legacy_answer_images(answer)
    if not isinstance(plan, dict) or not plan.get("show_images"):
        return text

    text = _filter_answer_paragraphs(text)
    if not text.strip():
        return synthesize_image_grounded_answer(query, plan)

    if _contains_pattern(text, _NOT_FOUND_PATTERNS):
        return synthesize_image_grounded_answer(query, plan)

    return text


@dataclass
class ImagePlannerToolSession:
    query: str
    doc_scope: list[str]
    server_ip: str
    cached_candidates: list[dict[str, Any]]

    def tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        arguments = arguments or {}
        if name == "search_image_candidates":
            doc_scope = arguments.get("doc_scope") or self.doc_scope
            normalized_scope = {
                normalize_doc_base_name(doc_name)
                for doc_name in doc_scope
                if str(doc_name or "").strip()
            }
            top_k = max(1, min(6, int(arguments.get("top_k") or 5)))
            candidates = []
            for item in self.cached_candidates:
                doc_name = str(item.get("doc_name") or "").strip()
                if normalized_scope and normalize_doc_base_name(doc_name) not in normalized_scope:
                    continue
                candidates.append(
                    {
                        "doc_name": doc_name,
                        "figure_key": item.get("figure_key"),
                        "caption": item.get("caption", ""),
                        "rank": int(item.get("rank") or 0),
                        "score": round(float(item.get("score") or 0.0), 6),
                        "fusion_score": round(float(item.get("fusion_score") or item.get("score") or 0.0), 6),
                        "image_score": round(float(item.get("image_score") or 0.0), 6),
                        "caption_score": round(float(item.get("caption_score") or 0.0), 6),
                        "semantic_role": str(item.get("semantic_role") or _infer_semantic_role(item)),
                        "visual_type": str(item.get("visual_type") or _caption_visual_type(item.get("caption", ""))),
                        "doc_rank": int(item.get("doc_rank") or 0),
                        "figure_label_match": bool(item.get("figure_label_match")),
                        "asset_relpath": item.get("asset_relpath", ""),
                    }
                )
            return json.dumps(
                {
                    "question": self.query,
                    "doc_scope": list(dict.fromkeys(doc_scope or self.doc_scope)),
                    "candidates": candidates[:top_k],
                },
                ensure_ascii=False,
            )

        if name == "get_figure_metadata":
            doc_name = str(arguments.get("doc_name") or "").strip()
            figure_key = str(arguments.get("figure_key") or "").strip()
            for rank, item in enumerate(self.cached_candidates, start=1):
                if item.get("doc_name") == doc_name and item.get("figure_key") == figure_key:
                    return json.dumps(
                        {
                            "doc_name": doc_name,
                            "figure_key": figure_key,
                            "caption": item.get("caption", ""),
                            "score": round(float(item.get("score") or 0.0), 6),
                            "fusion_score": round(float(item.get("fusion_score") or item.get("score") or 0.0), 6),
                            "image_score": round(float(item.get("image_score") or 0.0), 6),
                            "caption_score": round(float(item.get("caption_score") or 0.0), 6),
                            "semantic_role": str(item.get("semantic_role") or _infer_semantic_role(item)),
                            "visual_type": str(item.get("visual_type") or _caption_visual_type(item.get("caption", ""))),
                            "doc_rank": int(item.get("doc_rank") or 0),
                            "figure_label_match": bool(item.get("figure_label_match")),
                            "image_url": build_image_url(self.server_ip, doc_name, figure_key),
                            "display_caption": _build_display_caption(item.get("caption", ""), figure_key),
                            "rank": rank,
                        },
                        ensure_ascii=False,
                    )
            return json.dumps({"error": "figure_not_found"}, ensure_ascii=False)

        return json.dumps({"error": "unsupported_tool"}, ensure_ascii=False)


def build_image_planner_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_image_candidates",
                "description": "Search image candidates only within the already retrieved document scope for the current question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "doc_scope": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["question", "doc_scope", "top_k"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_figure_metadata",
                "description": "Get metadata for a specific candidate figure already returned by search_image_candidates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_name": {"type": "string"},
                        "figure_key": {"type": "string"},
                    },
                    "required": ["doc_name", "figure_key"],
                },
            },
        },
    ]


def _validate_and_hydrate_plan(
    raw_plan: dict[str, Any],
    allowed_candidates: list[dict[str, Any]],
    server_ip: str,
    planner_source: str = "planner",
) -> dict[str, Any]:
    if not raw_plan:
        return _build_default_plan("invalid_render_plan")

    show_images = bool(raw_plan.get("show_images", False))
    mode = str(raw_plan.get("mode") or "none").strip().lower()
    if mode not in _RENDER_MODES:
        mode = "none"

    try:
        confidence = float(raw_plan.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    reason_code = str(raw_plan.get("reason_code") or "").strip() or "planner_decision"

    allowed_map = {
        (str(item.get("doc_name") or "").strip(), str(item.get("figure_key") or "").strip()): item
        for item in allowed_candidates
        if str(item.get("doc_name") or "").strip() and str(item.get("figure_key") or "").strip()
    }
    selected_items = []
    seen = set()
    for raw_item in raw_plan.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        key = (
            str(raw_item.get("doc_name") or "").strip(),
            str(raw_item.get("figure_key") or "").strip(),
        )
        if not key[0] or not key[1] or key in seen or key not in allowed_map:
            continue
        seen.add(key)
        selected_items.append(key)

    if not show_images or confidence < _MIN_CONFIDENCE:
        return _build_default_plan(reason_code or "planner_declined")

    limit = _MODE_LIMITS[mode]
    if mode == "none":
        return _build_default_plan(reason_code)
    if mode == "compare" and len(selected_items) < 2:
        return _build_default_plan("insufficient_compare_items")
    if mode in {"single", "gallery", "compare"} and not selected_items:
        return _build_default_plan("no_valid_selected_items")

    selected_items = selected_items[:limit]
    hydrated_items = []
    for key in selected_items:
        candidate = allowed_map[key]
        hydrated_items.append(
            {
                "doc_name": key[0],
                "figure_key": key[1],
                "image_url": build_image_url(server_ip, key[0], key[1]),
                "caption": candidate.get("caption", ""),
                "display_caption": _build_display_caption(candidate.get("caption", ""), key[1]),
                "rank": int(candidate.get("rank") or 0),
            }
        )

    if mode == "single" and len(hydrated_items) > 1:
        hydrated_items = hydrated_items[:1]
    if mode == "gallery":
        hydrated_items = hydrated_items[:3]
    if mode == "compare":
        hydrated_items = hydrated_items[:2]

    if not hydrated_items:
        return _build_default_plan("no_renderable_items")

    if mode == "compare" and len(hydrated_items) != 2:
        return _build_default_plan("invalid_compare_selection")

    return {
        "show_images": True,
        "mode": mode,
        "presentation": _PRESENTATION_BY_MODE[mode],
        "items": hydrated_items,
        "reason_code": reason_code,
        "confidence": confidence,
        "planner_source": planner_source,
    }


def _parse_answer_planner_response(
    raw_plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    server_ip: str,
) -> dict[str, Any]:
    if not raw_plan:
        return {
            "answer_text": "",
            "render_plan": _build_default_plan("invalid_render_plan"),
        }

    render_payload = {
        "show_images": raw_plan.get("show_images", False),
        "mode": raw_plan.get("mode", "none"),
        "items": raw_plan.get("items", []),
        "reason_code": raw_plan.get("reason_code", ""),
        "confidence": raw_plan.get("confidence", 0.0),
    }
    return {
        "answer_text": str(raw_plan.get("answer_text") or "").strip(),
        "render_plan": _validate_and_hydrate_plan(render_payload, candidates, server_ip, planner_source="planner"),
    }


def _resolve_answer_language(query: str, preferred_language: str | None = None) -> str:
    if _is_chinese_text(query):
        return "中文"

    normalized = str(preferred_language or "").strip().lower()
    if normalized in {"chinese", "zh", "zh-cn", "中文", "cn"}:
        return "中文"
    if normalized in {"english", "en", "en-us"}:
        return "英文"
    return "与用户当前问题相同的语言"


def _build_answer_planner_prompt(
    query: str,
    doc_scope: list[str],
    evidence_summary: str,
    candidate_summary: str,
    mode_hint: str = "none",
    preferred_language: str | None = None,
) -> tuple[str, str]:
    answer_language = _resolve_answer_language(query, preferred_language)
    system_prompt = (
        "你是一个多模态回答规划器，用于检索增强对话。\n"
        "你的任务是同时决定是否展示图片，并生成最终回答文本。\n"
        "在输出最终结果前，你必须恰好调用一次 search_image_candidates。\n"
        "只有在认真考虑某张图时，才可以调用 get_figure_metadata。\n"
        "最终只允许输出 JSON，不要输出 Markdown，不要输出 HTML，不要输出解释性前后缀。\n"
        "JSON schema 固定为："
        "{\"answer_text\": string, "
        "\"show_images\": boolean, "
        "\"mode\": \"none\"|\"single\"|\"compare\"|\"gallery\", "
        "\"items\": [{\"doc_name\": string, \"figure_key\": string}], "
        "\"reason_code\": string, "
        "\"confidence\": number}。\n"
        "规则：\n"
        "- answer_text 必须严格基于已检索到的文本证据和工具返回结果。\n"
        "- 如果某张已选图片可以直接支撑答案，就在 answer_text 中明确说明这张图展示了什么。\n"
        "- 当 show_images=true 时，绝不能说“无法展示图片”“无法配图”或同类表述。\n"
        "- 当已检索证据或已选图片已经可以回答问题时，绝不能说“知识库中未找到答案”或同类表述。\n"
        "- 只能从工具返回的候选图里选择，不能编造新的 doc_name、figure_key 或图片链接。\n"
        "- 绝不允许跨文档选图；只允许在当前文档范围与候选 allowlist 内选择。\n"
        "- mode 必须服从候选语义与问题意图，不允许为了凑图而扩展到无关图。\n"
        "- 你必须综合考虑候选图的 score、rank、gap_to_next、visual_type 和 caption，不能只看 caption 文本表面语义。\n"
        "- 你必须综合考虑 fusion_score、image_score、caption_score、semantic_role、doc_rank、figure_label_match。\n"
        "- score 是主信号，rank 是次信号，caption 只用于解释为什么相关，不能单独决定选图。\n"
        "- gap_to_next 很大时，说明 top1 明显领先；gap_to_next 很小时，说明候选接近，不能轻易只选单张。\n"
        "- visual_type=ui_screenshot 或 code_or_terminal 的候选，除非用户明确要求，否则不要优先选择。\n"
        "- single 必须恰好选择 1 张图。\n"
        "- compare 必须恰好选择 2 张不同的图。\n"
        "- gallery 最多选择 3 张图。\n"
        f"- 当前 gate 已放行的 mode_hint 是 {mode_hint}；如果它不是 none，你必须遵守它，不能改成其他模式。\n"
        "- 对宽泛解释题，不要勉强选择单张局部图片；如果没有明显最佳图，可以选择 compare、gallery，或者直接 none。\n"
        "- 对明确 Figure N / 图N 的问题：优先看 figure 编号匹配，其次再看 score 优势。\n"
        "- 对宽泛解释题：如果 top1 没有明显领先，优先考虑 compare、gallery，或者 none，而不是机械选择 single。\n"
        "- 对 compare 类问题：只有当前两名都足够相关、且确实有对比价值时才用 compare；否则降级为 single 或 none。\n"
        "- 如果候选整体分数偏低，或者最相关图也只是局部细节图，宁可返回 none。\n"
        "- 回答语言默认必须使用"
        f"{answer_language}"
        "；只有当用户明确要求切换语言时，才允许使用其他语言。\n"
    )
    user_prompt = (
        f"用户问题：\n{str(query or '').strip()}\n\n"
        f"当前文档范围：\n{json.dumps(list(dict.fromkeys(doc_scope or [])), ensure_ascii=False)}\n\n"
        f"gate 放行模式：\n{mode_hint}\n\n"
        f"已检索文本证据摘要：\n{evidence_summary or '[empty]'}\n\n"
        f"当前图片候选摘要（按相关性从高到低排序）：\n{candidate_summary or '[empty]'}\n\n"
        "现在只输出最终 JSON。"
    )
    return system_prompt, user_prompt


def _build_gate_fallback_plan(query: str, gate: dict[str, Any], candidates: list[dict[str, Any]], server_ip: str) -> dict[str, Any]:
    mode_hint = str(gate.get("mode_hint") or "none").strip().lower()
    if mode_hint not in _RENDER_MODES or mode_hint == "none":
        return _build_default_plan(gate.get("reason_code") or "gate_declined")

    limit = _MODE_LIMITS[mode_hint]
    selected = [dict(item) for item in (candidates or [])[:limit]]
    if mode_hint == "compare" and len(selected) < 2:
        return _build_default_plan("insufficient_compare_items")
    if not selected:
        return _build_default_plan("no_gate_fallback_candidates")

    payload = {
        "show_images": True,
        "mode": mode_hint,
        "items": [
            {"doc_name": str(item.get("doc_name") or "").strip(), "figure_key": str(item.get("figure_key") or "").strip()}
            for item in selected
        ],
        "reason_code": str(gate.get("reason_code") or "gate_fallback"),
        "confidence": max(_MIN_CONFIDENCE, min(0.95, float(gate.get("top1_score") or 0.0))),
    }
    return _validate_and_hydrate_plan(payload, selected, server_ip, planner_source="gate_fallback")


def _run_answer_planner(
    planner_mdl,
    llm_setting: dict[str, Any],
    query: str,
    kbinfos: dict[str, Any],
    doc_scope: list[str],
    server_ip: str,
    preferred_language: str | None = None,
) -> dict[str, Any]:
    candidates = []
    for rank, item in enumerate((kbinfos or {}).get("image_candidates") or [], start=1):
        copied = dict(item)
        copied["rank"] = rank
        candidates.append(copied)

    if not str(query or "").strip() or not candidates:
        return {
            "answer_text": "",
            "render_plan": _build_default_plan("no_image_candidates"),
        }
    if not getattr(planner_mdl, "is_tools", False):
        return {
            "answer_text": "",
            "render_plan": _build_default_plan("planner_tools_unavailable"),
        }

    tool_session = ImagePlannerToolSession(
        query=str(query or "").strip(),
        doc_scope=list(dict.fromkeys(doc_scope or [])),
        server_ip=server_ip,
        cached_candidates=candidates,
    )
    planner_mdl.bind_tools(tool_session, build_image_planner_tools())

    evidence_summary = _build_evidence_summary(kbinfos)
    candidate_summary = _build_candidate_summary(candidates)
    system_prompt, user_prompt = _build_answer_planner_prompt(
        query,
        doc_scope,
        evidence_summary,
        candidate_summary,
        mode_hint=str(((kbinfos or {}).get("image_gate") or {}).get("mode_hint") or "none"),
        preferred_language=preferred_language,
    )

    planner_conf = dict(llm_setting or {})
    planner_conf["temperature"] = 0.1
    planner_conf["tool_choice"] = "auto"
    planner_conf["max_tokens"] = min(int(planner_conf.get("max_tokens", 768) or 768), 768)

    raw_plan = {}
    for attempt in range(2):
        try:
            raw_response = planner_mdl.chat(system_prompt, [{"role": "user", "content": user_prompt}], planner_conf)
            raw_plan = _parse_json_object(raw_response)
            if raw_plan:
                break
        except Exception:
            LOGGER.exception("Answer planner failed on attempt %s", attempt + 1)

    parsed = _parse_answer_planner_response(raw_plan, candidates, server_ip)
    gate_mode_hint = str((((kbinfos or {}).get("image_gate") or {}).get("mode_hint")) or "none").strip().lower()
    if (
        parsed["render_plan"].get("show_images")
        and gate_mode_hint in _RENDER_MODES
        and gate_mode_hint != "none"
        and parsed["render_plan"].get("mode") != gate_mode_hint
    ):
        parsed["render_plan"] = _build_default_plan("planner_mode_mismatch")
    if parsed["render_plan"].get("show_images"):
        return parsed

    gate = (kbinfos or {}).get("image_gate") or {}
    fallback_plan = _build_gate_fallback_plan(query, gate, candidates, server_ip)
    if fallback_plan.get("show_images"):
        answer_text = parsed.get("answer_text") or synthesize_image_grounded_answer(query, fallback_plan)
        return {
            "answer_text": answer_text,
            "render_plan": fallback_plan,
        }
    return parsed


def plan_image_rendering(
    planner_mdl,
    llm_setting: dict[str, Any],
    query: str,
    kbinfos: dict[str, Any],
    doc_scope: list[str],
    server_ip: str,
    preferred_language: str | None = None,
) -> dict[str, Any]:
    return _run_answer_planner(
        planner_mdl,
        llm_setting,
        query,
        kbinfos,
        doc_scope,
        server_ip,
        preferred_language=preferred_language,
    )["render_plan"]


def plan_answer_and_render(
    planner_mdl,
    llm_setting: dict[str, Any],
    query: str,
    kbinfos: dict[str, Any],
    doc_scope: list[str],
    server_ip: str,
    preferred_language: str | None = None,
) -> dict[str, Any]:
    result = _run_answer_planner(
        planner_mdl,
        llm_setting,
        query,
        kbinfos,
        doc_scope,
        server_ip,
        preferred_language=preferred_language,
    )
    render_plan = result["render_plan"]
    answer_text = postprocess_answer_with_render_plan(result.get("answer_text", ""), query, render_plan)
    return {
        "answer_text": answer_text,
        "render_plan": render_plan,
    }
