#!/usr/bin/env python3
"""
IR Benchmark — compare retrieval methods without re-indexing.

Dense search is done in-memory so you can test any embedding model
against the same OpenSearch BM25 index, no re-indexing required.

Methods compared:
  BM25        — OpenSearch multi_match (keyword baseline)
  Dense       — in-memory cosine similarity with any fastembed model
  Hybrid-RRF  — Reciprocal Rank Fusion of BM25 + Dense

Metrics: Hit@1, Hit@5, Hit@10, MRR@10, NDCG@10

Setup:
  pip install opensearch-py fastembed numpy

Usage:
  # Default model (same as production)
  python scripts/benchmark.py

  # Try a different model — no re-indexing needed
  python scripts/benchmark.py --model intfloat/multilingual-e5-small
  python scripts/benchmark.py --model BAAI/bge-m3

  # Save queries so the whole team runs the same test set
  python scripts/benchmark.py --save-queries queries.json

  # Load a saved test set
  python scripts/benchmark.py --load-queries queries.json

  # Remote OpenSearch
  python scripts/benchmark.py --opensearch-url http://my-server:9200
"""

import argparse
import json
import math
import os
import random
import re
import sys
from typing import Optional

import numpy as np
from fastembed import TextEmbedding
from opensearchpy import OpenSearch

INDEX_NAME = "books"
DEFAULT_K = [1, 5, 10]


# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_at_k(ranked: list[str], relevant: str, k: int) -> bool:
    return relevant in ranked[:k]

def mrr(ranked: list[str], relevant: str, k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id == relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg(ranked: list[str], relevant: str, k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id == relevant:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ── BM25 via OpenSearch ───────────────────────────────────────────────────────

def bm25_search(client: OpenSearch, query: str, k: int) -> list[str]:
    resp = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "synopsis", "authors^2", "genres^1.5", "publisher"],
                    "fuzziness": "AUTO",
                }
            },
            "_source": ["book_id"],
        },
    )
    return [h["_source"]["book_id"] for h in resp["hits"]["hits"]]


# ── In-memory Dense Search ────────────────────────────────────────────────────
# Embeds the corpus once, then answers queries with cosine similarity.
# No re-indexing in OpenSearch needed when swapping models.

class InMemoryIndex:
    def __init__(self, book_ids: list[str], matrix: np.ndarray):
        self.book_ids = book_ids
        self.matrix = matrix  # shape: (n_books, dim), L2-normalised

    def search(self, query_vector: np.ndarray, k: int) -> list[str]:
        scores = self.matrix @ query_vector          # cosine similarity
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [self.book_ids[i] for i in top_k]


def build_in_memory_index(books: list[dict], embedder: TextEmbedding) -> InMemoryIndex:
    texts = [f"{b['title']}. {b['synopsis']}" for b in books]
    book_ids = [b["book_id"] for b in books]

    print(f"  Embedding {len(texts)} books with {embedder.model_name}…")
    vectors = list(embedder.embed(texts))
    matrix = np.array(vectors, dtype=np.float32)

    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    return InMemoryIndex(book_ids, matrix)


# ── Hybrid RRF ────────────────────────────────────────────────────────────────

def hybrid_rrf(
    bm25_ids: list[str],
    dense_ids: list[str],
    k: int,
    rrf_k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]


# ── Query generation (heuristic, no LLM needed) ───────────────────────────────

def _parse_synopsis(synopsis: str) -> dict:
    """Parse the seeder's synopsis: '<Title> by <Authors>. Published in <Year>. Subjects: <X>.'"""
    authors, year, subjects = [], None, []
    m = re.search(r" by (.+?)\. Published", synopsis)
    if m:
        authors = [a.strip() for a in m.group(1).split(",")]
    m = re.search(r"Published in (\d{4})", synopsis)
    if m:
        year = m.group(1)
    m = re.search(r"Subjects?: (.+?)\.?\s*$", synopsis, re.I)
    if m:
        subjects = [s.strip() for s in m.group(1).split(",") if s.strip()]
    return {"authors": authors, "year": year, "subjects": subjects}


def _make_keyword_query(book: dict) -> Optional[str]:
    """2-3 words from the title. Easy for BM25, hard for dense."""
    words = [w for w in book.get("title", "").split() if len(w) > 3]
    if len(words) < 2:
        return None
    return " ".join(random.sample(words, min(3, len(words))))

def _make_author_query(book: dict) -> Optional[str]:
    """'books by <Author>' — tests author field boost."""
    p = _parse_synopsis(book.get("synopsis", ""))
    if not p["authors"]:
        return None
    return f"books by {p['authors'][0]}"

def _make_subject_query(book: dict) -> Optional[str]:
    """Subject as a standalone query — tests semantic retrieval."""
    p = _parse_synopsis(book.get("synopsis", ""))
    if not p["subjects"]:
        return None
    return random.choice(p["subjects"][:3])

def _make_year_subject_query(book: dict) -> Optional[str]:
    """'<subject> from <year>' — mixed fields query."""
    p = _parse_synopsis(book.get("synopsis", ""))
    if not p["year"] or not p["subjects"]:
        return None
    return f"{p['subjects'][0]} from {p['year']}"


STYLES = {
    "keyword": ("Title keywords",    _make_keyword_query),
    "author":  ("Author-based",      _make_author_query),
    "subject": ("Subject/conceptual",_make_subject_query),
    "year":    ("Year + subject",    _make_year_subject_query),
}


def generate_queries(books: list[dict], n: int, styles: list[str]) -> list[dict]:
    random.shuffle(books)
    queries = []
    style_keys = [s for s in styles if s in STYLES]

    for book in books:
        if len(queries) >= n:
            break
        style = random.choice(style_keys)
        q = STYLES[style][1](book)
        if q:
            queries.append({
                "query": q,
                "relevant_id": book["book_id"],
                "style": style,
                "title": book.get("title", ""),
            })

    if not queries:
        print("[!] Could not generate queries — is the index populated?")
        sys.exit(1)

    return queries


# ── Fetch books from OpenSearch ────────────────────────────────────────────────

def fetch_books(client: OpenSearch, limit: int = 2000) -> list[dict]:
    resp = client.search(
        index=INDEX_NAME,
        body={
            "size": 500,
            "query": {"match_all": {}},
            "_source": ["book_id", "title", "synopsis"],
        },
        scroll="2m",
    )
    books = [h["_source"] for h in resp["hits"]["hits"]]
    scroll_id = resp.get("_scroll_id")

    while scroll_id and len(books) < limit:
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        hits = resp["hits"]["hits"]
        if not hits:
            break
        books.extend(h["_source"] for h in hits)
        scroll_id = resp.get("_scroll_id")

    try:
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return [b for b in books if b.get("book_id") and b.get("title")][:limit]


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(
    queries: list[dict],
    client: OpenSearch,
    mem_index: InMemoryIndex,
    embedder: TextEmbedding,
    k_values: list[int],
) -> dict:
    max_k = max(k_values)
    acc = {
        m: {"hits": {k: 0 for k in k_values}, "mrr": [], "ndcg": []}
        for m in ("BM25", "Dense", "Hybrid-RRF")
    }

    for i, q in enumerate(queries):
        text, rel = q["query"], q["relevant_id"]
        print(f"  [{i+1:>3}/{len(queries)}] {text!r:<50}", end="\r")

        # Embed query once
        q_vec = np.array(list(embedder.query_embed(text))[0], dtype=np.float32)
        q_vec /= (np.linalg.norm(q_vec) or 1)

        bm25_ids  = bm25_search(client, text, max_k)
        dense_ids = mem_index.search(q_vec, max_k)
        hybrid_ids = hybrid_rrf(bm25_ids, dense_ids, max_k)

        for method, ids in (("BM25", bm25_ids), ("Dense", dense_ids), ("Hybrid-RRF", hybrid_ids)):
            for k in k_values:
                if hit_at_k(ids, rel, k):
                    acc[method]["hits"][k] += 1
            acc[method]["mrr"].append(mrr(ids, rel))
            acc[method]["ndcg"].append(ndcg(ids, rel))

    print()
    n = len(queries)
    return {
        method: {
            **{f"Hit@{k}": v["hits"][k] / n for k in k_values},
            "MRR@10":  sum(v["mrr"])  / n,
            "NDCG@10": sum(v["ndcg"]) / n,
        }
        for method, v in acc.items()
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_table(metrics: dict, model: str, k_values: list[int]) -> None:
    cols = [f"Hit@{k}" for k in k_values] + ["MRR@10", "NDCG@10"]
    col_w = 10
    header = f"{'Method':<14}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"Model : {model}")
    print(sep)
    print(header)
    print(sep)
    for method, scores in metrics.items():
        row = f"{method:<14}" + "".join(f"{scores[c]:>{col_w}.4f}" for c in cols)
        print(row)
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BM25 vs Dense vs Hybrid-RRF without re-indexing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--opensearch-url", default=os.getenv("OPENSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
                        help="fastembed model name (default: BAAI/bge-small-en-v1.5)")
    parser.add_argument("--n", type=int, default=100, help="Number of test queries (default: 100)")
    parser.add_argument("--styles", nargs="+", choices=list(STYLES.keys()), default=list(STYLES.keys()))
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_K, metavar="K")
    parser.add_argument("--save-queries", metavar="FILE", help="Save generated queries to JSON")
    parser.add_argument("--load-queries", metavar="FILE", help="Load queries from JSON")
    args = parser.parse_args()

    # connect
    from urllib.parse import urlparse
    p = urlparse(args.opensearch_url)
    client = OpenSearch(
        hosts=[{"host": p.hostname, "port": p.port or 9200}],
        http_compress=True,
        use_ssl=p.scheme == "https",
    )
    if not client.ping():
        print(f"[!] Cannot reach OpenSearch at {args.opensearch_url}")
        sys.exit(1)

    if not client.indices.exists(index=INDEX_NAME) or client.count(index=INDEX_NAME)["count"] == 0:
        print(f"[!] Index '{INDEX_NAME}' is empty. Run POST /index/bulk first.")
        sys.exit(1)

    n_indexed = client.count(index=INDEX_NAME)["count"]
    print(f"[+] OpenSearch: {n_indexed} books in index")

    # queries
    if args.load_queries:
        queries = json.load(open(args.load_queries))
        print(f"[+] Loaded {len(queries)} queries from {args.load_queries}")
    else:
        books = fetch_books(client, limit=max(args.n * 5, 500))
        print(f"[+] Fetched {len(books)} books for query generation")
        queries = generate_queries(books, args.n, args.styles)
        print(f"[+] Generated {len(queries)} queries")
        if args.save_queries:
            json.dump(queries, open(args.save_queries, "w"), indent=2, ensure_ascii=False)
            print(f"[+] Saved queries → {args.save_queries}")

    style_counts = {}
    for q in queries:
        style_counts[q["style"]] = style_counts.get(q["style"], 0) + 1
    for style, count in style_counts.items():
        print(f"    {STYLES[style][0]:<28} {count} queries")

    # load model and build in-memory index
    print(f"[+] Loading model: {args.model}")
    embedder = TextEmbedding(model_name=args.model)

    # fetch books for in-memory index (only need title + synopsis text)
    all_books = fetch_books(client, limit=n_indexed)
    mem_index = build_in_memory_index(all_books, embedder)

    # run
    print(f"[+] Evaluating…")
    metrics = run_eval(queries, client, mem_index, embedder, args.k)
    print_table(metrics, args.model, args.k)


if __name__ == "__main__":
    main()
