#!/usr/bin/env bash
# One-time script to apply live OpenSearch index tuning without reindexing.
# Run after deploying Phase 1: bash scripts/update_index_settings.sh

OPENSEARCH="${OPENSEARCH_URL:-http://localhost:9200}"
INDEX="books"

echo "==> Updating refresh_interval to 5s (reduces I/O during heavy indexing)..."
curl -s -XPUT "$OPENSEARCH/$INDEX/_settings" \
  -H "Content-Type: application/json" \
  -d '{"index": {"refresh_interval": "5s"}}' | python3 -m json.tool

echo ""
echo "==> Current index settings:"
curl -s "$OPENSEARCH/$INDEX/_settings" | python3 -m json.tool

echo ""
echo "==> Cluster health:"
curl -s "$OPENSEARCH/_cluster/health?pretty"
