"""
Optional ONNX cross-encoder reranker (ms-marco-MiniLM-L-6-v2, ~23 MB).

Disabled by default (ENABLE_RERANKER=false). When enabled, reranks the top
RERANKER_TOP_K RRF candidates using a cross-encoder — adds ~50-150ms on CPU
but improves result quality for ambiguous queries.

Requires: fastembed[reranker] in requirements.txt
"""

import os
import threading
from typing import Any

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in {"1", "true", "yes", "on"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))

_reranker = None
_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from fastembed.reranking.cross_encoder import TextCrossEncoder
                _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    candidates: list[tuple[str, float, dict[str, Any]]],
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Rerank candidates using cross-encoder scores.

    candidates: list of (book_id, rrf_score, source_dict)
    Returns the same structure sorted by cross-encoder score (desc),
    with candidates beyond RERANKER_TOP_K appended unchanged.
    """
    if not ENABLE_RERANKER or not candidates:
        return candidates

    try:
        r = _get_reranker()
        top = candidates[:RERANKER_TOP_K]
        rest = candidates[RERANKER_TOP_K:]

        passages = [
            f"{c[2].get('title', '')}. {c[2].get('synopsis', '')}"
            for c in top
        ]
        scores = list(r.rerank(query, passages))

        reranked = sorted(
            zip([c[0] for c in top], scores, [c[2] for c in top]),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(reranked) + rest
    except Exception:
        return candidates
