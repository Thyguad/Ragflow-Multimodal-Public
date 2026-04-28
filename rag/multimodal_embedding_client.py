import base64
import logging
import mimetypes
import os
import threading
import time
from pathlib import Path

from dashscope import MultiModalEmbedding
from dashscope.embeddings.multimodal_embedding import (
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
)


LOGGER = logging.getLogger(__name__)


class Qwen3VLEmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
        max_retries: int = 2,
        min_interval_seconds: float = 0.25,
    ):
        self.api_key = (api_key or os.getenv("RAGFLOW_IMAGE_EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        self.model_name = (
            model_name
            or os.getenv("RAGFLOW_IMAGE_EMBEDDING_MODEL")
            or "qwen3-vl-embedding"
        ).strip()
        self.dimension = int(
            dimension
            or os.getenv("RAGFLOW_IMAGE_EMBEDDING_DIMENSION")
            or 1024
        )
        self.max_retries = max(0, int(max_retries))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def is_configured(self) -> bool:
        return bool(self.api_key and self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_caption(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_image(self, image_path_or_bytes) -> list[float]:
        image_payload = self._as_data_uri(image_path_or_bytes)
        item = MultiModalEmbeddingItemImage(image=image_payload, factor=1.0)
        return self._call([item])

    def embed_fusion(self, text: str, image_path_or_bytes) -> list[float]:
        image_payload = self._as_data_uri(image_path_or_bytes)
        items = [
            MultiModalEmbeddingItemText(text=str(text or "").strip(), factor=1.0),
            MultiModalEmbeddingItemImage(image=image_payload, factor=1.0),
        ]
        return self._call(items, enable_fusion=True)

    def _embed_text(self, text: str) -> list[float]:
        item = MultiModalEmbeddingItemText(text=str(text or "").strip(), factor=1.0)
        return self._call([item])

    def _call(self, items, **kwargs) -> list[float]:
        if not self.is_configured():
            raise ValueError("DashScope multimodal embedding client is not configured.")

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                response = MultiModalEmbedding.call(
                    model=self.model_name,
                    input=items,
                    api_key=self.api_key,
                    dimension=self.dimension,
                    **kwargs,
                )
                return self._extract_embedding(response)
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Qwen3-VL embedding call failed on attempt %s/%s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise last_error

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.time()
            delta = now - self._last_call_ts
            if delta < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - delta)
            self._last_call_ts = time.time()

    def _extract_embedding(self, response) -> list[float]:
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output", {})
        embeddings = {}
        if isinstance(output, dict):
            embeddings = output
        elif output is not None:
            embeddings = dict(output)
        items = embeddings.get("embeddings") or []
        if not items:
            raise ValueError("DashScope multimodal embedding response did not include embeddings.")
        vector = items[0].get("embedding") if isinstance(items[0], dict) else None
        if not vector:
            raise ValueError("DashScope multimodal embedding response did not include an embedding vector.")
        if len(vector) != self.dimension:
            raise ValueError(f"Expected {self.dimension} dimensions, got {len(vector)}.")
        return [float(value) for value in vector]

    def _as_data_uri(self, image_path_or_bytes) -> str:
        mime_type = "image/jpeg"
        payload = b""
        if isinstance(image_path_or_bytes, (str, os.PathLike, Path)):
            image_path = Path(image_path_or_bytes)
            payload = image_path.read_bytes()
            mime_type = mimetypes.guess_type(str(image_path))[0] or mime_type
        elif isinstance(image_path_or_bytes, bytes):
            payload = image_path_or_bytes
        else:
            raise TypeError("image_path_or_bytes must be a path or bytes.")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
