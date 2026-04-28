from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

FIGURE_REF_PATTERN = re.compile(r"!\[(.*?)\]\((p\d+_\d+)\)")
FIGURE_PLACEHOLDER_PATTERN = re.compile(r"!\[FIG_PLACEHOLDER\]\(#\)")

def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources exist for tokenization used by RAGFlow.
    Some environments miss `punkt_tab`, which will crash chunking.
    """
    try:
        import nltk
    except Exception:
        return

    # Prefer project venv directories (so no need for global install).
    candidates = [
        os.getenv("NLTK_DATA", ""),
        str(ROOT_DIR / ".venv" / "nltk_data"),
        str(ROOT_DIR / ".venv" / "share" / "nltk_data"),
        str(ROOT_DIR / ".venv" / "lib" / "nltk_data"),
    ]
    for p in candidates:
        if p and p not in nltk.data.path:
            nltk.data.path.append(p)

    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    ]
    for path, pkg in required:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # If download fails (e.g., offline), let downstream raise a clear error.
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly test the VLM parsing chain without starting the full RAGFlow app."
    )
    parser.add_argument("file", help="Path to the input document (pdf/docx/pptx).")
    parser.add_argument(
        "--mode",
        choices=("parser", "naive", "both"),
        default="both",
        help="Which part of the VLM chain to run.",
    )
    parser.add_argument("--from-page", type=int, default=0, help="Start page index, 0-based.")
    parser.add_argument(
        "--to-page",
        type=int,
        default=3,
        help="End page index, exclusive. Defaults to parsing the first 3 pages.",
    )
    parser.add_argument(
        "--lang",
        default="Chinese",
        help="Language passed to rag.app.naive.chunk(). Default: Chinese.",
    )
    parser.add_argument(
        "--chunk-token-num",
        type=int,
        default=128,
        help="Chunk size used by rag.app.naive.chunk().",
    )
    parser.add_argument(
        "--delimiter",
        default="\n!?。；！？",
        help="Delimiter used by rag.app.naive.chunk().",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory used to store markdown, figures, chunk summaries and debug artifacts.",
    )
    parser.add_argument("--vlm-api-key", default="", help="Override RAGFLOW_VLM_API_KEY.")
    parser.add_argument("--vlm-api-url", default="", help="Override RAGFLOW_VLM_API_URL.")
    parser.add_argument("--vlm-modelid", default="", help="Override RAGFLOW_VLM_MODEL.")
    parser.add_argument("--markdown-api-key", default="", help="Override RAGFLOW_MD_API_KEY.")
    parser.add_argument("--markdown-api-url", default="", help="Override RAGFLOW_MD_API_URL.")
    parser.add_argument("--markdown-modelid", default="", help="Override RAGFLOW_MD_MODEL.")
    return parser.parse_args()


def detect_file_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".docx", ".doc"}:
        return "word"
    if suffix in {".ppt", ".pptx"}:
        return "ppt"
    raise ValueError(f"Unsupported file type: {suffix}")


def build_api_config(args: argparse.Namespace) -> dict[str, str]:
    vlm_api_key = args.vlm_api_key or os.getenv("RAGFLOW_VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    vlm_api_url = args.vlm_api_url or os.getenv("RAGFLOW_VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    vlm_modelid = args.vlm_modelid or os.getenv("RAGFLOW_VLM_MODEL", "qwen2.5-vl-72b-instruct")

    markdown_api_key = args.markdown_api_key or os.getenv("RAGFLOW_MD_API_KEY") or vlm_api_key
    markdown_api_url = args.markdown_api_url or os.getenv("RAGFLOW_MD_API_URL") or vlm_api_url
    markdown_modelid = args.markdown_modelid or os.getenv("RAGFLOW_MD_MODEL", "qwen-plus")

    return {
        "vlm_api_key": vlm_api_key,
        "vlm_api_url": vlm_api_url,
        "vlm_modelid": vlm_modelid,
        "markdown_api_key": markdown_api_key,
        "markdown_api_url": markdown_api_url,
        "markdown_modelid": markdown_modelid,
    }


def validate_api_config(api_config: dict[str, str]) -> None:
    missing = [key for key in ("vlm_api_key", "markdown_api_key") if not api_config.get(key)]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing API config: {missing_str}. "
            "Please pass CLI arguments or set RAGFLOW_VLM_API_KEY / RAGFLOW_MD_API_KEY."
        )


def prepare_output_dir(file_path: Path, output_dir: str) -> Path:
    if output_dir:
        out_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT_DIR / "logs" / "vlm_chain" / f"{file_path.stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def configure_nltk_data() -> None:
    nltk_dir = ROOT_DIR / "nltk_data"
    if not nltk_dir.exists():
        return

    existing = os.environ.get("NLTK_DATA", "")
    paths = [str(nltk_dir)]
    if existing:
        paths.append(existing)
    os.environ["NLTK_DATA"] = os.pathsep.join(paths)

    try:
        import nltk

        nltk_path = str(nltk_dir)
        if nltk_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_path)
    except Exception:
        return


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_image(path: Path, image: Any) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save extracted images.") from exc

    if isinstance(image, Image.Image):
        image.save(path, format="JPEG")
        return
    if isinstance(image, bytes):
        path.write_bytes(image)
        return
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def resolve_placeholders(
    markdown_text: str,
    figures_dict: dict[str, bytes],
    captions_dict: dict[str, str],
    figure_order: list[str | None] | None = None,
) -> str:
    figure_keys = figure_order or list(figures_dict.keys())
    index = 0

    def replace_placeholder(_: re.Match[str]) -> str:
        nonlocal index
        while index < len(figure_keys) and not figure_keys[index]:
            index += 1
        if index >= len(figure_keys):
            return ""
        key = figure_keys[index]
        index += 1
        if key not in figures_dict:
            return ""
        caption = captions_dict.get(key, "示意图")
        return f"![{caption}]({key})"

    resolved = FIGURE_PLACEHOLDER_PATTERN.sub(replace_placeholder, markdown_text)
    return re.sub(r"\n{3,}", "\n\n", resolved).strip()


def write_parser_outputs(
    out_dir: Path,
    markdown_text: str,
    figures_dict: dict[str, bytes],
    captions_dict: dict[str, str],
    figure_order: list[str | None] | None,
) -> dict[str, Any]:
    # 确保 parser_mode 输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_md_path = out_dir / "parser_markdown_raw.md"
    raw_md_path.write_text(markdown_text, encoding="utf-8")

    resolved_markdown = resolve_placeholders(markdown_text, figures_dict, captions_dict, figure_order)
    resolved_md_path = out_dir / "parser_markdown_resolved.md"
    resolved_md_path.write_text(resolved_markdown, encoding="utf-8")

    figures_dir = out_dir / "parser_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for key, img_bytes in figures_dict.items():
        (figures_dir / f"{key}.jpg").write_bytes(img_bytes)

    captions_path = out_dir / "parser_captions.json"
    save_json(captions_path, captions_dict)

    placeholder_count = len(FIGURE_PLACEHOLDER_PATTERN.findall(markdown_text))
    resolved_ref_count = len(FIGURE_REF_PATTERN.findall(resolved_markdown))

    summary = {
        "raw_markdown_path": str(raw_md_path),
        "resolved_markdown_path": str(resolved_md_path),
        "captions_path": str(captions_path),
        "figures_dir": str(figures_dir),
        "figure_count": len(figures_dict),
        "caption_count": len(captions_dict),
        "figure_order_count": len(figure_order or []),
        "placeholder_count": placeholder_count,
        "resolved_reference_count": resolved_ref_count,
        "figure_keys": list(figures_dict.keys()),
    }
    save_json(out_dir / "parser_summary.json", summary)
    return summary


def run_parser_mode(
    file_path: Path,
    file_type: str,
    api_config: dict[str, str],
    from_page: int,
    to_page: int,
    out_dir: Path,
) -> dict[str, Any]:
    from deepdoc.parser.vlm_doc_parser import VLMDocParser

    binary = file_path.read_bytes()
    parser_out_dir = out_dir / "parser_mode"
    parser_api_config = dict(api_config)
    parser_api_config["debug_dir"] = str(parser_out_dir / "layout_debug")
    parser = VLMDocParser(api_config=parser_api_config)
    markdown_text, figures_dict, captions_dict = parser(
        binary=binary,
        file_type=file_type,
        from_page=from_page,
        to_page=to_page,
    )
    summary = write_parser_outputs(
        out_dir=parser_out_dir,
        markdown_text=markdown_text,
        figures_dict=figures_dict,
        captions_dict=captions_dict,
        figure_order=getattr(parser, "figure_order", None),
    )
    summary["layout_debug_dir"] = str(parser_out_dir / "layout_debug")
    save_json(parser_out_dir / "parser_summary.json", summary)
    return summary


def serialize_chunk(chunk_data: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in chunk_data.items():
        if key == "image":
            continue
        serialized[key] = value
    return serialized


def run_naive_mode(
    file_path: Path,
    binary: bytes,
    api_config: dict[str, str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    configure_nltk_data()
    from rag.app.naive import chunk

    callback_events: list[dict[str, Any]] = []

    def callback(progress=None, msg="") -> None:
        event = {"progress": progress, "message": msg}
        callback_events.append(event)
        prefix = f"[{progress}]" if progress is not None else "[msg]"
        print(f"{prefix} {msg}")

    parser_config = {
        "layout_recognize": "VLM",
        "chunk_token_num": args.chunk_token_num,
        "delimiter": args.delimiter,
        "vlm_config": api_config,
    }
    chunks = chunk(
        filename=file_path.name,
        binary=binary,
        from_page=args.from_page,
        to_page=args.to_page,
        lang=args.lang,
        callback=callback,
        parser_config=parser_config,
    )

    naive_dir = out_dir / "naive_mode"
    naive_dir.mkdir(parents=True, exist_ok=True)
    image_dir = naive_dir / "chunk_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    serialized_chunks = []
    content_preview_lines = []
    image_chunk_count = 0
    doc_type_counter = Counter()

    for index, chunk_data in enumerate(chunks, start=1):
        serialized = serialize_chunk(chunk_data)
        image = chunk_data.get("image")
        if image is not None:
            image_chunk_count += 1
            image_filename = f"chunk_{index:03d}.jpg"
            save_image(image_dir / image_filename, image)
            serialized["debug_image_file"] = image_filename
        doc_type = serialized.get("doc_type_kwd") or serialized.get("doc_type") or "text"
        doc_type_counter[doc_type] += 1
        serialized_chunks.append(serialized)

        content = serialized.get("content_with_weight", "")
        preview = str(content).strip().replace("\n", " ")
        content_preview_lines.append(f"## Chunk {index}\n\n{preview[:800]}\n")

    save_json(naive_dir / "chunks.json", serialized_chunks)
    save_json(naive_dir / "callback_events.json", callback_events)
    (naive_dir / "chunks_preview.md").write_text("\n".join(content_preview_lines), encoding="utf-8")

    summary = {
        "chunks_path": str(naive_dir / "chunks.json"),
        "preview_path": str(naive_dir / "chunks_preview.md"),
        "callback_events_path": str(naive_dir / "callback_events.json"),
        "chunk_image_dir": str(image_dir),
        "chunk_count": len(serialized_chunks),
        "image_chunk_count": image_chunk_count,
        "doc_type_counts": dict(doc_type_counter),
    }
    save_json(naive_dir / "naive_summary.json", summary)
    return summary


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    args = parse_args()
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return 1

    try:
        file_type = detect_file_type(file_path)
        api_config = build_api_config(args)
        validate_api_config(api_config)
    except Exception as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    out_dir = prepare_output_dir(file_path, args.output_dir)
    binary = file_path.read_bytes()

    # Make sure tokenizers are available before running naive mode.
    if args.mode in {"naive", "both"}:
        ensure_nltk_resources()

    run_meta = {
        "file": str(file_path),
        "file_type": file_type,
        "mode": args.mode,
        "from_page": args.from_page,
        "to_page": args.to_page,
        "lang": args.lang,
        "chunk_token_num": args.chunk_token_num,
        "delimiter": args.delimiter,
        "output_dir": str(out_dir),
        "vlm_api_url": api_config["vlm_api_url"],
        "vlm_modelid": api_config["vlm_modelid"],
        "markdown_api_url": api_config["markdown_api_url"],
        "markdown_modelid": api_config["markdown_modelid"],
    }
    save_json(out_dir / "run_meta.json", run_meta)

    try:
        if args.mode in {"parser", "both"}:
            parser_summary = run_parser_mode(
                file_path=file_path,
                file_type=file_type,
                api_config=api_config,
                from_page=args.from_page,
                to_page=args.to_page,
                out_dir=out_dir,
            )
            print_summary("Parser Mode Summary", parser_summary)

        if args.mode in {"naive", "both"}:
            naive_summary = run_naive_mode(
                file_path=file_path,
                binary=binary,
                api_config=api_config,
                args=args,
                out_dir=out_dir,
            )
            print_summary("Naive Mode Summary", naive_summary)
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
