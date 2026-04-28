import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.image_asset_utils import (
    get_doc_captions_path,
    get_doc_figures_dir,
    get_image_output_root,
    load_captions_dict,
)
from rag.image_index import ensure_doc_image_index, load_doc_image_index
from rag.multimodal_embedding_client import Qwen3VLEmbeddingClient


DEFAULT_DOCS: list[str] = []
DEFAULT_OUTPUT = Path("data/eval_samples/image_eval/ablation_report.json")
DEFAULT_CACHE_DIR = Path("data/eval_samples/image_eval/ablation_cache")

GENERIC_CAPTION_PATTERNS = [
    re.compile(r"^(?:figure|fig\.?|图)\s*[:：]?\s*\d+[\.\-:：\s]*$", re.IGNORECASE),
    re.compile(r"^(?:module|chip)\s+appearance\b", re.IGNORECASE),
]


@dataclass
class FigureEntry:
    doc_name: str
    figure_key: str
    caption: str
    image_path: Path


@dataclass
class EvalQuery:
    qid: str
    question: str
    doc_name: str
    figure_key: str
    topic: str


def _normalize_vector(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr if norm <= 0 else arr / norm


def _strip_caption_prefix(caption: str) -> str:
    text = str(caption or "").strip()
    text = re.sub(r"^(?:figure|fig\.?|图)\s*[:：]?\s*\d+[\.\-:：\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\([a-z0-9]+\)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;：，。")
    return text


def _is_meaningful_topic(topic: str) -> bool:
    text = str(topic or "").strip()
    if len(text) < 6:
        return False
    lowered = text.lower()
    if "二维码" in text or "qr code" in lowered:
        return False
    if lowered in {"function diagram", "system model", "framework"}:
        return False
    return not any(pattern.match(text) for pattern in GENERIC_CAPTION_PATTERNS)


def _build_question(topic: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", topic):
        return f"哪张图展示了{topic}？请直接展示图片。"
    return f"Which figure shows {topic}? Please show the image."


def _cache_key(doc_name: str, figure_key: str, mode: str, dimension: int) -> str:
    payload = f"{doc_name}::{figure_key}::{mode}::{dimension}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cached_vector(cache_dir: Path, cache_key: str):
    path = cache_dir / f"{cache_key}.npy"
    if not path.is_file():
        return None
    return np.load(path)


def _write_cached_vector(cache_dir: Path, cache_key: str, vector: np.ndarray):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{cache_key}.npy", np.asarray(vector, dtype=np.float32))


def _collect_entries(doc_names: list[str], client: Qwen3VLEmbeddingClient):
    entries = []
    image_vecs = []
    for doc_name in doc_names:
        ensure_doc_image_index(doc_name, client=client)
        manifest, vectors, _caption_vecs = load_doc_image_index(doc_name)
        captions = load_captions_dict(doc_name)
        figures_dir = get_doc_figures_dir(doc_name)
        if not manifest or vectors.size == 0:
            continue
        entry_map = {(e["doc_name"], e["figure_key"]): e for e in manifest.get("entries") or []}
        for idx, vector in enumerate(vectors):
            entry = manifest["entries"][idx]
            caption = captions.get(entry["figure_key"], entry.get("caption") or "")
            image_path = figures_dir / f"{entry['figure_key']}.jpg"
            if not image_path.is_file():
                continue
            entries.append(
                FigureEntry(
                    doc_name=entry["doc_name"],
                    figure_key=entry["figure_key"],
                    caption=caption,
                    image_path=image_path,
                )
            )
            image_vecs.append(np.asarray(vector, dtype=np.float32))
    return entries, np.asarray(image_vecs, dtype=np.float32)


def _discover_doc_names() -> list[str]:
    output_root = get_image_output_root()
    if not output_root.is_dir():
        return []
    doc_names = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir() or path.name in {"groundtruth", "test"}:
            continue
        if get_doc_captions_path(path.name).is_file() and get_doc_figures_dir(path.name).is_dir():
            doc_names.append(path.name)
    return doc_names


def _build_eval_queries(entries: list[FigureEntry], max_queries_per_doc: int):
    queries = []
    per_doc = {}
    for entry in entries:
        topic = _strip_caption_prefix(entry.caption)
        if not _is_meaningful_topic(topic):
            continue
        doc_count = per_doc.get(entry.doc_name, 0)
        if doc_count >= max_queries_per_doc:
            continue
        qid = f"{entry.doc_name}:{entry.figure_key}"
        queries.append(
            EvalQuery(
                qid=qid,
                question=_build_question(topic),
                doc_name=entry.doc_name,
                figure_key=entry.figure_key,
                topic=topic,
            )
        )
        per_doc[entry.doc_name] = doc_count + 1
    return queries


def _cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.dot(matrix, vector)


def _evaluate_mode(name: str, entries: list[FigureEntry], queries: list[EvalQuery], score_fn):
    doc_to_indices = {}
    for idx, entry in enumerate(entries):
        doc_to_indices.setdefault(entry.doc_name, []).append(idx)

    def eval_scope(scope_name: str, scoped_indices_fn):
        hit1 = hit3 = 0
        mrr = 0.0
        rows = []
        for query in queries:
            indices = scoped_indices_fn(query)
            scores = score_fn(query, indices)
            ranked_local = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
            ranked = [entries[idx] for idx, _ in ranked_local]
            top3 = ranked[:3]
            found_rank = None
            for rank, entry in enumerate(ranked, start=1):
                if entry.doc_name == query.doc_name and entry.figure_key == query.figure_key:
                    found_rank = rank
                    break
            if found_rank == 1:
                hit1 += 1
            if found_rank is not None and found_rank <= 3:
                hit3 += 1
            if found_rank is not None:
                mrr += 1.0 / found_rank
            rows.append(
                {
                    "qid": query.qid,
                    "question": query.question,
                    "target": {"doc_name": query.doc_name, "figure_key": query.figure_key},
                    "top3": [
                        {
                            "doc_name": entry.doc_name,
                            "figure_key": entry.figure_key,
                            "caption": entry.caption,
                        }
                        for entry in top3
                    ],
                    "rank": found_rank,
                }
            )
        total = max(1, len(queries))
        return {
            "scope": scope_name,
            "queries": len(queries),
            "hit@1": hit1 / total,
            "hit@3": hit3 / total,
            "mrr": mrr / total,
            "details": rows,
        }

    return {
        "mode": name,
        "doc_scoped": eval_scope("doc_scoped", lambda query: doc_to_indices.get(query.doc_name, [])),
        "global_pool": eval_scope("global_pool", lambda query: list(range(len(entries)))),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare image-only, late-fusion, and pure-fusion retrieval.")
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS, help="Document names to evaluate. Defaults to discovered parser outputs.")
    parser.add_argument("--max-queries-per-doc", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--late-fusion-caption-weights", nargs="*", type=float, default=[0.4, 0.6, 0.8])
    args = parser.parse_args()

    client = Qwen3VLEmbeddingClient()
    if not client.is_configured():
        raise SystemExit("Qwen3-VL embedding is not configured.")

    cache_dir = Path(args.cache_dir)
    doc_names = list(dict.fromkeys(args.docs or _discover_doc_names()))
    if not doc_names:
        raise SystemExit("No documents configured. Pass --docs or run VLM parsing first.")
    entries, image_vecs = _collect_entries(doc_names, client)
    queries = _build_eval_queries(entries, args.max_queries_per_doc)
    if not entries or not queries:
        raise SystemExit("No entries or queries available for ablation.")

    caption_vecs = []
    fusion_vecs = []
    for entry in entries:
        caption_cache_key = _cache_key(entry.doc_name, entry.figure_key, "caption", client.dimension)
        fusion_cache_key = _cache_key(entry.doc_name, entry.figure_key, "fusion", client.dimension)

        caption_vec = _load_cached_vector(cache_dir, caption_cache_key)
        if caption_vec is None:
            caption_vec = _normalize_vector(client.embed_caption(entry.caption))
            _write_cached_vector(cache_dir, caption_cache_key, caption_vec)
        caption_vecs.append(caption_vec)

        fusion_vec = _load_cached_vector(cache_dir, fusion_cache_key)
        if fusion_vec is None:
            fusion_vec = _normalize_vector(client.embed_fusion(entry.caption, entry.image_path))
            _write_cached_vector(cache_dir, fusion_cache_key, fusion_vec)
        fusion_vecs.append(fusion_vec)

    caption_vecs = np.asarray(caption_vecs, dtype=np.float32)
    fusion_vecs = np.asarray(fusion_vecs, dtype=np.float32)

    query_vecs = {}
    for query in queries:
        query_vecs[query.qid] = _normalize_vector(client.embed_query(query.question))

    results = []

    results.append(
        _evaluate_mode(
            "image_only",
            entries,
            queries,
            lambda query, indices: _cosine_scores(image_vecs[indices], query_vecs[query.qid]),
        )
    )

    best_late_fusion = None
    for caption_weight in args.late_fusion_caption_weights:
        image_weight = 1.0 - caption_weight
        result = _evaluate_mode(
            f"late_fusion_caption_{caption_weight:.1f}",
            entries,
            queries,
            lambda query, indices, iw=image_weight, cw=caption_weight: (
                iw * _cosine_scores(image_vecs[indices], query_vecs[query.qid])
                + cw * _cosine_scores(caption_vecs[indices], query_vecs[query.qid])
            ),
        )
        results.append(result)
        if best_late_fusion is None or result["doc_scoped"]["hit@1"] > best_late_fusion["doc_scoped"]["hit@1"]:
            best_late_fusion = result

    results.append(
        _evaluate_mode(
            "pure_fusion",
            entries,
            queries,
            lambda query, indices: _cosine_scores(fusion_vecs[indices], query_vecs[query.qid]),
        )
    )

    summary = {
        "docs": doc_names,
        "query_count": len(queries),
        "queries": [
            {
                "qid": query.qid,
                "question": query.question,
                "topic": query.topic,
                "target": {"doc_name": query.doc_name, "figure_key": query.figure_key},
            }
            for query in queries
        ],
        "results": results,
        "best_late_fusion": None if best_late_fusion is None else best_late_fusion["mode"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
