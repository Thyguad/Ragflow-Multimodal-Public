import hashlib
import json
import os
from pathlib import Path


IMAGE_OUTPUT_ROOT_ENV = "RAGFLOW_IMAGE_OUTPUT_ROOT"
DEFAULT_OUTPUT_DIRNAME = "image_assets"
DEFAULT_RUN_ID = "default"
IMAGE_RETRIEVAL_PARSER_CONFIG_KEY = "enable_multimodal_image_retrieval"
IMAGE_INDEX_VERSION = "v2_image_caption_late_fusion"
IMAGE_FUSION_IMAGE_WEIGHT = 0.4
IMAGE_FUSION_CAPTION_WEIGHT = 0.6


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_image_output_root(output_root: str | os.PathLike | None = None) -> Path:
    root = output_root or os.getenv(IMAGE_OUTPUT_ROOT_ENV)
    if root:
        return Path(root).expanduser().resolve()
    return (get_repo_root() / DEFAULT_OUTPUT_DIRNAME).resolve()


def ensure_image_output_root(output_root: str | os.PathLike | None = None) -> Path:
    root = get_image_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def derive_run_id(output_root: str | os.PathLike | None = None) -> str:
    root = get_image_output_root(output_root)
    if root.name == DEFAULT_OUTPUT_DIRNAME:
        return DEFAULT_RUN_ID
    if root.parent.name == "output_runs":
        return root.name
    return root.name or DEFAULT_RUN_ID


def normalize_doc_base_name(doc_name: str) -> str:
    return os.path.splitext(str(doc_name or "").strip())[0]


def get_doc_asset_dir(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_image_output_root(output_root) / normalize_doc_base_name(doc_name)


def get_doc_figures_dir(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_asset_dir(doc_name, output_root) / "figures"


def get_doc_captions_path(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_asset_dir(doc_name, output_root) / "captions.json"


def get_doc_manifest_path(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_asset_dir(doc_name, output_root) / "manifest.json"


def get_doc_image_index_manifest_path(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_asset_dir(doc_name, output_root) / "image_index_manifest.json"


def get_doc_image_index_vectors_path(doc_name: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_asset_dir(doc_name, output_root) / "image_index_vectors.npz"


def get_figure_path(doc_name: str, figure_key: str, output_root: str | os.PathLike | None = None) -> Path:
    return get_doc_figures_dir(doc_name, output_root) / f"{figure_key}.jpg"


def build_asset_relpath(doc_name: str, figure_key: str) -> str:
    return f"{normalize_doc_base_name(doc_name)}/figures/{figure_key}.jpg"


def build_logical_image_ref(doc_name: str, figure_key: str, caption: str) -> str:
    return f"<image_assets/{normalize_doc_base_name(doc_name)}/figures/{figure_key}:{caption}>"


def build_image_url(server_ip: str, doc_name: str, figure_key: str) -> str:
    return f"{server_ip.rstrip('/')}/image_assets/{normalize_doc_base_name(doc_name)}/figures/{figure_key}.jpg"


def iter_image_doc_names(output_root: str | os.PathLike | None = None) -> list[str]:
    root = get_image_output_root(output_root)
    if not root.is_dir():
        return []
    return [
        child.name
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir() and (child / "captions.json").is_file()
    ]


def load_captions_dict(doc_name: str, output_root: str | os.PathLike | None = None) -> dict[str, str]:
    captions_path = get_doc_captions_path(doc_name, output_root)
    if not captions_path.is_file():
        return {}
    try:
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(captions, dict):
        return {}
    return captions


def load_json_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_sha256_bytes(data: bytes | None) -> str:
    if not data:
        return ""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: str | os.PathLike | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def captions_hash(captions: dict[str, str]) -> str:
    normalized = {
        str(key): str(value or "").strip()
        for key, value in sorted((captions or {}).items(), key=lambda item: item[0])
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return compute_sha256_bytes(encoded)


def is_multimodal_image_retrieval_enabled(parser_config: dict | None) -> bool:
    if not isinstance(parser_config, dict):
        return False
    return bool(parser_config.get(IMAGE_RETRIEVAL_PARSER_CONFIG_KEY, False))
