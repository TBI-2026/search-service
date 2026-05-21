"""Standalone entrypoint for the RabbitMQ consumer process."""

import os

from dotenv import load_dotenv
from opensearchpy import OpenSearch

import consumer
from embedding import build_embedder_from_env

load_dotenv()

os_client = OpenSearch(
    hosts=[{
        "host": os.getenv("OPENSEARCH_HOST", "opensearch-node"),
        "port": int(os.getenv("OPENSEARCH_PORT", 9200)),
    }],
    http_compress=True,
    use_ssl=False,
)
model = build_embedder_from_env()

consumer.start(os_client, model)
