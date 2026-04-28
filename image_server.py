from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from rag.image_asset_utils import ensure_image_output_root

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_image_output_root()

app = Flask(__name__)


@app.get("/")
def read_root():
    return jsonify(
        {
            "message": "RAGFlow image server is online",
            "output_dir": str(OUTPUT_DIR),
            "status": "online",
        }
    )


@app.get("/image_assets/<path:filepath>")
def serve_output(filepath: str):
    return send_from_directory(str(OUTPUT_DIR), filepath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
