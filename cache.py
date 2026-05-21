"""
Redis-backed cache for query embeddings and search results.

All public functions are safe to call even when Redis is unavailable —
exceptions are caught and treated as cache misses so search never breaks.
"""

import hashlib
import json
import os

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis-search")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
EMBEDDING_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", 3600))
RESULTS_TTL = int(os.getenv("RESULTS_CACHE_TTL", 300))

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def _emb_key(text: str) -> str:
    return "emb:" + hashlib.sha256(text.encode()).hexdigest()


def _res_key(q: str, limit: int, threshold: float) -> str:
    raw = f"{q}|{limit}|{threshold}"
    return "res:" + hashlib.sha256(raw.encode()).hexdigest()


def get_embedding(text: str) -> list[float] | None:
    try:
        raw = get_client().get(_emb_key(text))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def set_embedding(text: str, vector: list[float]) -> None:
    try:
        get_client().setex(_emb_key(text), EMBEDDING_TTL, json.dumps(vector))
    except Exception:
        pass


def get_results(q: str, limit: int, threshold: float) -> list | None:
    try:
        raw = get_client().get(_res_key(q, limit, threshold))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def set_results(q: str, limit: int, threshold: float, results: list) -> None:
    try:
        get_client().setex(_res_key(q, limit, threshold), RESULTS_TTL, json.dumps(results))
    except Exception:
        pass
