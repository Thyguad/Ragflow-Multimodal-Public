import re

from rag.image_asset_utils import normalize_doc_base_name


DOCUMENT_TITLE_PATTERN = re.compile(r"[《“\"]([^》”\"]{2,200})[》”\"]")


def _extract_explicit_document_titles(query: str) -> list[str]:
    titles = []
    for match in DOCUMENT_TITLE_PATTERN.finditer(str(query or "")):
        title = normalize_doc_base_name(match.group(1))
        if title and title not in titles:
            titles.append(title)
    return titles


def _collect_candidate_doc_names(kbinfos) -> list[str]:
    names = []
    for doc in (kbinfos or {}).get("doc_aggs", []):
        base_name = normalize_doc_base_name(doc.get("doc_name") if isinstance(doc, dict) else doc)
        if base_name:
            names.append(base_name)

    for chunk in (kbinfos or {}).get("chunks", []):
        base_name = normalize_doc_base_name(
            chunk.get("doc_name") or chunk.get("docnm_kwd") or chunk.get("document_name")
        )
        if base_name:
            names.append(base_name)

    seen = set()
    ordered = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def select_image_candidate_doc_names(kbinfos, query: str = "", max_docs: int = 3) -> list[str]:
    ordered = _collect_candidate_doc_names(kbinfos)
    if not ordered:
        return []

    query_text = str(query or "").strip()
    if not query_text:
        return ordered[: max(1, int(max_docs or 3))]

    explicit_titles = _extract_explicit_document_titles(query_text)
    if explicit_titles:
        title_matches = []
        for name in ordered:
            base_name = normalize_doc_base_name(name)
            if any(base_name == title or title in base_name for title in explicit_titles):
                title_matches.append(name)
        if title_matches:
            return title_matches[: max(1, int(max_docs or 3))]

    explicit_matches = []
    lowered_query = query_text.lower()
    for name in ordered:
        base_name = normalize_doc_base_name(name)
        if base_name and base_name.lower() in lowered_query:
            explicit_matches.append(name)

    if explicit_matches:
        return explicit_matches[: max(1, int(max_docs or 3))]

    return ordered[: max(1, int(max_docs or 3))]
