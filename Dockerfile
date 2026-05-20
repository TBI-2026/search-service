FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Warm model cache to reduce first-request latency.
RUN python -c "from fastembed import TextEmbedding; m=TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); next(m.embed(['warmup']))"

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
