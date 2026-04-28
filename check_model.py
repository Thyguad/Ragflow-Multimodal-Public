import os
from pathlib import Path

p = Path(__file__).resolve().parent / "rag" / "res" / "deepdoc" / "det.onnx"
print("exists:", os.path.exists(p))
if os.path.exists(p):
    print("size:", os.path.getsize(p))
