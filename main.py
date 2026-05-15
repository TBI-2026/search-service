"""
search-service: Hybrid semantic search for FondasiKehidupan books.

Endpoints
---------
GET  /search          - hybrid BM25 + kNN search
POST /index           - index a single book document
POST /index/bulk      - bulk-index all books from PostgreSQL
GET  /health          - health check
"""

import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from opensearchpy import OpenSearch, helpers
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "opensearch-node")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "books"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "fondasikehidupan"),
    "user": os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# ---------------------------------------------------------------------------
# Shared singletons (initialised at startup)
# ---------------------------------------------------------------------------

os_client: OpenSearch = None
model: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global os_client, model

    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    ensure_index()

    import consumer as _consumer
    t = threading.Thread(target=_consumer.start, args=(os_client, model), daemon=True)
    t.start()

    yield


app = FastAPI(title="FondasiKehidupan Search Service", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

INDEX_MAPPING = {
    "settings": {
        "index.knn": True,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "book_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "standard", "boost": 2},
            "synopsis": {"type": "text", "analyzer": "standard"},
            "authors": {"type": "text", "analyzer": "standard"},
            "genres": {"type": "keyword"},
            "publisher": {"type": "text"},
            "synopsis_vector": {
                "type": "knn_vector",
                "dimension": VECTOR_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
    },
}


def ensure_index() -> None:
    if not os_client.indices.exists(index=INDEX_NAME):
        os_client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class BookDocument(BaseModel):
    book_id: str
    title: str
    synopsis: str
    authors: list[str] = []
    genres: list[str] = []
    publisher: str = ""


class SearchResult(BaseModel):
    book_id: str
    title: str
    score: float


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def build_text(title: str, synopsis: str) -> str:
    return f"{title}. {synopsis}"


def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_hits: list[dict],
    knn_hits: list[dict],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_source"]["book_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        details[doc_id] = hit["_source"]

    for rank, hit in enumerate(knn_hits):
        doc_id = hit["_source"]["book_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        details[doc_id] = hit["_source"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "opensearch": os_client.ping()}


@app.get("/search", response_model=list[SearchResult])
def search(
    q: str = Query(..., description="Natural language search query"),
    limit: int = Query(5, ge=1, le=50),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum RRF score filter (0 = no filter)"),
):
    query_vector = embed(q)
    fetch_k = max(limit * 4, 20)  # fetch more for RRF then trim

    # --- BM25 query ---
    bm25_resp = os_client.search(
        index=INDEX_NAME,
        body={
            "size": fetch_k,
            "query": {
                "multi_match": {
                    "query": q,
                    "fields": ["title^3", "synopsis", "authors^2", "genres^1.5", "publisher"],
                    "fuzziness": "AUTO",
                }
            },
            "_source": ["book_id", "title"],
        },
    )
    bm25_hits = bm25_resp["hits"]["hits"]

    # --- kNN query ---
    knn_resp = os_client.search(
        index=INDEX_NAME,
        body={
            "size": fetch_k,
            "query": {
                "knn": {
                    "synopsis_vector": {
                        "vector": query_vector,
                        "k": fetch_k,
                    }
                }
            },
            "_source": ["book_id", "title"],
        },
    )
    knn_hits = knn_resp["hits"]["hits"]

    ranked = reciprocal_rank_fusion(bm25_hits, knn_hits)

    results = []
    for book_id, score in ranked[:limit]:
        if threshold > 0 and score < threshold:
            continue
        # Retrieve title from cached hit data
        title = next(
            (h["_source"].get("title", "") for h in bm25_hits + knn_hits
             if h["_source"]["book_id"] == book_id),
            "",
        )
        results.append(SearchResult(book_id=book_id, title=title, score=round(score, 6)))

    return results


@app.post("/index", status_code=201)
def index_book(doc: BookDocument):
    text = build_text(doc.title, doc.synopsis)
    vector = embed(text)

    os_client.index(
        index=INDEX_NAME,
        id=doc.book_id,
        body={
            "book_id": doc.book_id,
            "title": doc.title,
            "synopsis": doc.synopsis,
            "authors": doc.authors,
            "genres": doc.genres,
            "publisher": doc.publisher,
            "synopsis_vector": vector,
        },
        refresh="wait_for",
    )
    return {"indexed": doc.book_id}


@app.post("/index/bulk")
def bulk_index():
    """Read all books from PostgreSQL and index them in OpenSearch."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT
                b.id AS book_id,
                b.title,
                b.synopsis,
                b.book_picture,
                p.name AS publisher,
                ARRAY_AGG(DISTINCT a.name) AS authors,
                ARRAY_AGG(DISTINCT g.genre) AS genres
            FROM book b
            JOIN publisher p ON p.id = b.id_publisher
            LEFT JOIN authored_by ab ON ab.id_book = b.id
            LEFT JOIN author a ON a.id = ab.id_author
            LEFT JOIN having_genre hg ON hg.id_book = b.id
            LEFT JOIN genre g ON g.id = hg.id_genre
            GROUP BY b.id, p.name
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    def generate_actions():
        for row in rows:
            text = build_text(row["title"], row["synopsis"])
            vector = embed(text)
            yield {
                "_index": INDEX_NAME,
                "_id": str(row["book_id"]),
                "_source": {
                    "book_id": str(row["book_id"]),
                    "title": row["title"],
                    "synopsis": row["synopsis"],
                    "authors": [a for a in (row["authors"] or []) if a],
                    "genres": [g for g in (row["genres"] or []) if g],
                    "publisher": row["publisher"] or "",
                    "synopsis_vector": vector,
                },
            }

    success, failed = helpers.bulk(os_client, generate_actions(), raise_on_error=False)
    return {"indexed": success, "failed": failed}
