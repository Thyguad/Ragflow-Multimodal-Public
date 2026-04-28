import argparse
import json
from pathlib import Path

from rag.image_asset_utils import get_image_output_root, iter_image_doc_names
from rag.image_index import backfill_doc_image_indexes


def main():
    parser = argparse.ArgumentParser(description="Backfill multimodal image sidecars from existing output assets.")
    parser.add_argument("--docs", nargs="*", default=None, help="Optional document names to backfill.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild sidecars even if they look current.")
    parser.add_argument("--output", default=None, help="Optional image output root. Defaults to RAGFLOW_IMAGE_OUTPUT_ROOT or ./image_assets.")
    args = parser.parse_args()

    output_root = get_image_output_root(args.output)
    doc_names = args.docs or iter_image_doc_names(output_root)
    results = backfill_doc_image_indexes(
        doc_names,
        output_root=output_root,
        force_rebuild=args.force_rebuild,
    )
    payload = {
        "output_root": str(output_root),
        "doc_count": len(doc_names),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
