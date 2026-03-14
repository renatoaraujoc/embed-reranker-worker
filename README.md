# Serverless Embedding + Reranking Worker

A single Docker image that serves embedding and reranking models on any GPU infrastructure. Supports **RunPod queue-based** serverless endpoints and **standard HTTP load balancer** mode (RunPod, Kubernetes, Cloud Run, ECS, or bare metal).

## Features

- **Dual mode** — auto-detects RunPod queue (serverless handler) or load balancer (FastAPI HTTP server) via the `PORT` env var
- **OpenAI-compatible** `/v1/embeddings` endpoint — works with `@ai-sdk/openai-compatible` and any OpenAI SDK client
- **Reranking** via `/v1/rerank` (load balancer) or RunPod's `/openai/v1/rerank` and `/runsync` (queue)
- **Flexible deployment** — run embed-only, rerank-only, or both on the same endpoint
- **Flash Attention 2** — auto-detected, with safe SDPA fallback
- **MRL truncation** — server-side default or client-override via `dimensions`; vectors are truncated and re-normalized
- **Lazy model loading** (queue mode) — models load on first request, `/v1/models` responds instantly
- **Eager model loading** (load balancer mode) — models load at startup, `/ping` returns 204 until ready then 200
- **Structured logging** — model loading, request routing, and execution times logged to stdout for RunPod's log viewer

## Model Compatibility

Any HuggingFace model compatible with `sentence-transformers` works out of the box:

- **Embedding**: any model that works with `SentenceTransformer` — BERT, Qwen3, NomicBERT, XLM-R, E5, BGE, GTE, etc.
- **Reranking**: any model that works with `CrossEncoder` — MiniLM, ModernBERT, XLM-R, BGE rerankers, etc.

`trust_remote_code=True` is enabled by default, so models with custom code (e.g., nomic-embed) work without extra configuration.

### Tested Models

| Model | Type | Notes |
|-------|------|-------|
| Qwen/Qwen3-Embedding-8B | Embedding | Decoder-based, up to 40K tokens, flash attention |
| BAAI/bge-reranker-v2-m3 | Reranking | Best multilingual open-source reranker |

## Environment Variables

### Shared

These apply to both embedding and reranking services.

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace access token. Required for gated models (e.g., Llama, Gemma). Not needed for public models. Passed automatically to `huggingface_hub` via the environment. |
| `EMBED_MODEL` | — | HuggingFace model ID for embeddings. At least one of `EMBED_MODEL` or `RERANK_MODEL` must be set. |
| `RERANK_MODEL` | — | HuggingFace model ID for reranking. At least one of `EMBED_MODEL` or `RERANK_MODEL` must be set. |
| `DTYPE` | `auto` | Torch dtype for model loading: `auto`, `float16`, `bfloat16`, `float32`. Use `float16` for large models to halve VRAM. |
| `USE_FLASH_ATTN` | `true` | Enable Flash Attention 2 if available. Falls back to SDPA automatically if unavailable. |
| `MAX_CONTEXT_LENGTH` | model default | Override max sequence length. Texts exceeding this are truncated by the tokenizer. |
| `MAX_CLIENT_BATCH_SIZE` | unlimited | Max number of texts (embed) or documents (rerank) per request. Rejects with error if exceeded. |
| `RUNPOD_MAX_CONCURRENCY` | `1` | Concurrent jobs per worker (queue mode only). |

### Embedding-specific

| Variable | Default | Client overridable | Description |
|----------|---------|-------------------|-------------|
| `EMBED_BATCH_SIZE` | `32` | No | GPU forward pass batch size. Set to `1` for max context length inputs to avoid OOM. |
| `EMBED_DEFAULT_DIMENSIONS` | native (no truncation) | Yes, via `dimensions` in request | Default MRL truncation dimensions. If not set, returns full native dimensions (e.g., 4096 for Qwen3). Client can override per-request by sending `dimensions` in the payload. |

**Client override example** — server has `EMBED_DEFAULT_DIMENSIONS=1024`, client sends `"dimensions": 512`:

```json
// Request — client overrides to 512
{ "model": "embedder", "input": ["text"], "dimensions": 512 }

// Response — 512 dimensions, truncated and re-normalized
{ "data": [{ "embedding": [0.01, ...], "index": 0 }] }
```

If neither server nor client sets dimensions, the model's native dimensions are returned (e.g., 4096 for Qwen3-Embedding-8B).

### Reranking-specific

| Variable | Default | Client overridable | Description |
|----------|---------|-------------------|-------------|
| `RERANK_BATCH_SIZE` | `32` | No | GPU forward pass batch size for query-document pairs. |

**Client-controlled parameters** (always per-request, no server-side default):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | all documents | Number of top results to return, sorted by relevance score descending. |
| `return_documents` | `true` | Include document text in response. Set to `false` to return only indices and scores. |

## API

### Embeddings — `POST /v1/embeddings`

**Request:**
```json
{
  "model": "embedder",
  "input": ["text1", "text2"],
  "dimensions": 1024
}
```

**Response (OpenAI format):**
```json
{
  "object": "list",
  "model": "Qwen/Qwen3-Embedding-8B",
  "data": [
    { "object": "embedding", "embedding": [0.01, -0.02, ...], "index": 0 },
    { "object": "embedding", "embedding": [0.03, 0.01, ...], "index": 1 }
  ],
  "usage": { "prompt_tokens": 12, "total_tokens": 12 }
}
```

### Reranking — `POST /v1/rerank`

**Request:**
```json
{
  "query": "What is environmental legislation?",
  "documents": ["doc1 text", "doc2 text", "doc3 text"],
  "top_n": 10,
  "return_documents": true
}
```

**Response:**
```json
{
  "results": [
    { "index": 0, "relevance_score": 0.999, "document": { "text": "doc1 text" } },
    { "index": 2, "relevance_score": 0.534, "document": { "text": "doc3 text" } }
  ],
  "usage": { "prompt_tokens": 42, "total_tokens": 42 }
}
```

### Models — `GET /v1/models`

Returns configured model IDs without loading models.

### Health — `GET /ping` (load balancer mode)

Returns `200` when ready, `204` while initializing.

## Deployment

### RunPod Queue Mode (serverless, scale-to-zero)

1. Create a **Queue-based** endpoint on RunPod
2. Set container image to `ghcr.io/renatoaraujoc/embed-reranker-worker:latest`
3. Set env vars (`EMBED_MODEL`, `RERANK_MODEL`, etc.)
4. Set container disk to **25 GB** (for model downloads)
5. Requests go through RunPod's `/openai/v1/embeddings` proxy or `/runsync`

### RunPod Load Balancer Mode (low latency, direct HTTP)

1. Create a **Load Balancer** endpoint on RunPod
2. Same image and env vars
3. Requests go directly to `https://{endpoint_id}.api.runpod.ai/v1/embeddings`
4. No queue overhead — lower latency for real-time inference

### Container Disk

The default 5 GB container disk is not enough for large models. Set **25 GB** for 8B parameter models, **10 GB** for smaller reranker-only deployments.

### FlashBoot

Enable FlashBoot on the endpoint for fast cold starts (~2s). After the first successful request, RunPod snapshots the container state including downloaded models.

### Example: Qwen3-Embedding-8B on 48 GB GPU

```
EMBED_MODEL=Qwen/Qwen3-Embedding-8B
DTYPE=float16
EMBED_BATCH_SIZE=1
MAX_CONTEXT_LENGTH=40960
MAX_CLIENT_BATCH_SIZE=128
EMBED_DEFAULT_DIMENSIONS=1024
```

### Example: bge-reranker-v2-m3 on 24 GB GPU

```
RERANK_MODEL=BAAI/bge-reranker-v2-m3
DTYPE=float16
RERANK_BATCH_SIZE=32
MAX_CLIENT_BATCH_SIZE=128
```

## Docker Image

Pre-built images are published to GitHub Container Registry on every tagged release:

```bash
docker pull ghcr.io/renatoaraujoc/embed-reranker-worker:latest
docker pull ghcr.io/renatoaraujoc/embed-reranker-worker:v1.0.8
```

### Building locally

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/renatoaraujoc/embed-reranker-worker:v1.0.8 .
```

### Releasing

Push a version tag to trigger the GitHub Actions build + release:

```bash
git tag v1.0.9
git push origin v1.0.9
```

## Stack

- **Runtime**: Python 3.11, CUDA 12.4
- **ML**: PyTorch 2.6.0, sentence-transformers 5.3.0, transformers 5.3.0
- **Acceleration**: Flash Attention 2.7.4 (prebuilt wheel, cxx11abiFALSE)
- **Serving**: RunPod SDK 1.8.1 (queue), FastAPI + Uvicorn (load balancer)

## Performance (RunPod, 48 GB GPU)

| Task | Model | Warm Latency |
|------|-------|-------------|
| Embedding (3 texts, 1024 dims) | Qwen3-Embedding-8B | ~100ms |
| Embedding (40K tokens, 1024 dims) | Qwen3-Embedding-8B | ~13s |
| Reranking (5 docs) | bge-reranker-v2-m3 | ~75ms |

## License

MIT
