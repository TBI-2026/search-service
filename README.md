# Search Service (Hybrid Retrieval)

`search-service` is a FastAPI-based hybrid retrieval service for book search.

## Features

- Hybrid ranking: **BM25 + vector kNN + Reciprocal Rank Fusion (RRF)**
- Async search path using `AsyncOpenSearch`
- Parallel BM25 and kNN query execution
- Redis cache for:
  - query embeddings
  - final search results
- Optional cross-encoder reranking (`ENABLE_RERANKER`)
- Single-document indexing (`POST /index`)
- Bulk indexing from PostgreSQL (`POST /index/bulk`)
- RabbitMQ consumer for `book.created` events
- Optional API key for indexing endpoints

## Runtime Architecture

Docker Compose runs three services:

- `search-service`: FastAPI API (`:8001`)
- `redis` (`redis-search`): caching layer
- `search-consumer`: standalone RabbitMQ consumer process

## API Endpoints

- `GET /health`
- `GET /search?q=<query>&limit=<1..50>&threshold=<0..1>`
- `POST /index`
- `POST /index/bulk`

## Example Requests

Search:

```bash
curl "http://localhost:8001/search?q=python%20backend&limit=5&threshold=0.02"
```

Index one document:

```bash
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  -H "X-Index-Api-Key: YOUR_KEY_IF_CONFIGURED" \
  -d '{
    "book_id":"c76d6d43-57da-4bfd-8ce8-53f1e20e44b1",
    "title":"Clean Architecture",
    "synopsis":"A practical guide to software architecture.",
    "authors":["Robert C. Martin"],
    "genres":["Software Engineering"],
    "publisher":"Prentice Hall",
    "bookPicture":"https://example.com/cover.jpg"
  }'
```

Bulk index from PostgreSQL:

```bash
curl -X POST "http://localhost:8001/index/bulk" \
  -H "X-Index-Api-Key: YOUR_KEY_IF_CONFIGURED"
```

## Environment Variables

Start from:

```bash
cp .env.example .env
```

Core:

- `OPENSEARCH_HOST`, `OPENSEARCH_PORT`
- `OPENSEARCH_REPLICAS`, `OPENSEARCH_REFRESH_INTERVAL`
- `EMBEDDING_MODEL`, `FASTEMBED_CACHE_PATH`
- `INDEX_API_KEY` (empty = disabled)
- `UVICORN_WORKERS`
- `ENABLE_CONSUMER`

Caching:

- `REDIS_HOST`, `REDIS_PORT`
- `EMBEDDING_CACHE_TTL`
- `RESULTS_CACHE_TTL`

Optional reranker:

- `ENABLE_RERANKER`
- `RERANKER_MODEL`
- `RERANKER_TOP_K`

Messaging + DB:

- `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASS`
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`

## Run with Docker

Make sure the shared network exists:

```bash
docker network create internal-network || true
```

Run stack:

```bash
docker compose up -d --build
```

Check status and logs:

```bash
docker compose ps
docker logs -f search-service
docker logs -f search-consumer
curl http://localhost:8001/health
```

## Search Flow

1. Check full-result cache by `(q, limit, threshold)`.
2. Check embedding cache by query text.
3. If cache miss, compute query embedding with ONNX model.
4. Execute BM25 and vector kNN in parallel.
5. Fuse ranks with RRF.
6. Optionally rerank top candidates with cross-encoder.
7. Return results and store them in Redis cache.

## Indexing Flow

Event-based indexing:

1. Backend publishes `book.created`.
2. `search-consumer` reads event from RabbitMQ.
3. Consumer embeds `title + synopsis`.
4. Consumer indexes document into OpenSearch.

Manual reindex:

- `POST /index/bulk` pulls all books from PostgreSQL and indexes in chunks.

## Current Scalability Defaults

- OpenSearch index:
  - `number_of_shards: 1`
  - `number_of_replicas: OPENSEARCH_REPLICAS` (default `0`)
- API workers:
  - `UVICORN_WORKERS` (default `2` in Docker command)
- Consumer is separated into its own process (`search-consumer`).
