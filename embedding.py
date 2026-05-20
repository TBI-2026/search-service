import math
import os
from typing import Iterable, List

from fastembed import TextEmbedding


class ONNXEmbedder:
    """Lightweight ONNX embedder wrapper used by API and consumer."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _normalize(vec: Iterable[float]) -> List[float]:
        values = [float(v) for v in vec]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            return values
        return [v / norm for v in values]

    def _embed_one(self, iterator):
        return self._normalize(next(iterator))

    def embed_document(self, text: str) -> List[float]:
        return self._embed_one(self._model.embed([text]))

    def embed_query(self, text: str) -> List[float]:
        # query_embed applies model-specific query prefixing when available.
        if hasattr(self._model, "query_embed"):
            return self._embed_one(self._model.query_embed(text))
        return self.embed_document(text)


def build_embedder_from_env() -> ONNXEmbedder:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/app/.cache/fastembed")
    return ONNXEmbedder(model_name=model_name, cache_dir=cache_dir)
