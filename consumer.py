"""
RabbitMQ consumer for the search-service.

Listens to the book.created routing key on wiwokdetok.exchange and indexes
each book into OpenSearch as soon as it is created.

Called from main.py via a background daemon thread:
    consumer.start(os_client, model)
"""

import json
import logging
import os
import time

import pika
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

EXCHANGE_NAME = "wiwokdetok.exchange"
QUEUE_NAME = "fondasikehidupan.search.queue"
ROUTING_KEY = "book.created"
INDEX_NAME = "books"

RABBITMQ_CONFIG = dict(
    host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=int(os.getenv("RABBITMQ_PORT", 5672)),
    virtual_host="/",
    credentials=pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "guest"),
        os.getenv("RABBITMQ_PASS", "guest"),
    ),
    heartbeat=600,
    blocked_connection_timeout=300,
)


def _make_callback(os_client, model):
    def callback(ch, method, properties, body):
        should_ack = False
        try:
            data = json.loads(body)
            book_id = data.get("id") or data.get("bookId")
            title = data.get("title", "")
            synopsis = data.get("synopsis", "")
            cover = data.get("bookPicture", "")

            if not book_id or not title:
                # Invalid payload; drop to avoid poison-message loop.
                should_ack = True
                return

            text = f"{title}. {synopsis}"
            vector = model.embed_document(text)

            os_client.index(
                index=INDEX_NAME,
                id=str(book_id),
                body={
                    "book_id": str(book_id),
                    "title": title,
                    "synopsis": synopsis,
                    "authors": [],
                    "genres": [],
                    "publisher": "",
                    "book_picture": cover,
                    "synopsis_vector": vector,
                },
            )
            log.info("Indexed book %s ('%s')", book_id, title)
            should_ack = True
        except Exception as exc:
            log.error("Failed to index book: %s", exc)
        finally:
            if should_ack:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Retry transient failures (OpenSearch/network spikes).
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    return callback


def start(os_client, model) -> None:
    """Run the consumer loop. Reconnects automatically on failure."""
    while True:
        try:
            params = pika.ConnectionParameters(**RABBITMQ_CONFIG)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.exchange_declare(
                exchange=EXCHANGE_NAME, exchange_type="topic", durable=True
            )
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.queue_bind(
                queue=QUEUE_NAME,
                exchange=EXCHANGE_NAME,
                routing_key=ROUTING_KEY,
            )
            channel.basic_qos(prefetch_count=10)
            channel.basic_consume(
                queue=QUEUE_NAME,
                on_message_callback=_make_callback(os_client, model),
            )

            log.info("RabbitMQ consumer started, waiting for book.created events…")
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            log.warning("RabbitMQ connection failed (%s). Retrying in 5s…", e)
            time.sleep(5)
        except Exception as e:
            log.error("Consumer error: %s. Restarting in 5s…", e)
            time.sleep(5)
