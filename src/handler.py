import os
import time
from config import Config
from embed_service import EmbedService
from rerank_service import RerankService
from logger import log

config = Config()
embed_service = EmbedService(config) if config.embed_model else None
rerank_service = RerankService(config) if config.rerank_model else None


def list_models():
    """Return configured model IDs without requiring models to be loaded."""
    data = []
    if config.embed_model:
        data.append({"id": config.embed_model, "object": "model", "type": "embedding"})
    if config.rerank_model:
        data.append({"id": config.rerank_model, "object": "model", "type": "reranker"})
    return {"object": "list", "data": data}


async def handler(job):
    job_input = job["input"]

    # /v1/models responds instantly — no model loading required
    if "openai_route" in job_input and job_input["openai_route"] == "/v1/models":
        return list_models()

    # Lazy model loading (first real request triggers GPU model load)
    if embed_service:
        await embed_service.ensure_loaded()
    if rerank_service:
        await rerank_service.ensure_loaded()

    # ── OpenAI-compatible routes (via RunPod /openai/ proxy) ──
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        payload = job_input.get("openai_input", {})

        if route == "/v1/embeddings":
            if not embed_service:
                return {"error": "Embeddings not configured (EMBED_MODEL not set)"}
            texts = payload.get("input", [])
            n = 1 if isinstance(texts, str) else len(texts)
            dims = payload.get('dimensions') or config.default_dimensions or 'native'
            log.info(f"[queue] /v1/embeddings — {n} texts, dims={dims}")
            start = time.perf_counter()
            result = embed_service.embed(payload)
            log.info(f"[queue] Embeddings complete — {(time.perf_counter() - start) * 1000:.0f}ms")
            return result

        if route == "/v1/rerank":
            if not rerank_service:
                return {"error": "Reranking not configured (RERANK_MODEL not set)"}
            log.info(f"[queue] /v1/rerank — {len(payload.get('documents', []))} docs")
            start = time.perf_counter()
            result = rerank_service.rerank(payload)
            log.info(f"[queue] Rerank complete — {(time.perf_counter() - start) * 1000:.0f}ms")
            return result

        return {"error": f"Unsupported route: {route}"}

    # ── Standard RunPod /runsync input (reranking) ──
    if "query" in job_input and "documents" in job_input:
        if not rerank_service:
            return {"error": "Reranking not configured (RERANK_MODEL not set)"}
        log.info(f"[queue] /runsync rerank — {len(job_input.get('documents', []))} docs")
        start = time.perf_counter()
        result = rerank_service.rerank(job_input)
        log.info(f"[queue] Rerank complete — {(time.perf_counter() - start) * 1000:.0f}ms")
        return result

    return {"error": "Invalid input: expected openai_route or query+documents"}


# ── Mode detection ──────────────────────────────────────────────────────
port = os.environ.get("PORT")

if port:
    # Load balancer mode — RunPod sets PORT, expects HTTP server
    import uvicorn
    from http_server import create_app

    log.info(f"Starting HTTP server (load balancer mode) on port {port}")
    log.info(f"  embed_model={config.embed_model or 'disabled'}")
    log.info(f"  rerank_model={config.rerank_model or 'disabled'}")
    app = create_app(config, embed_service, rerank_service)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
else:
    # Queue mode — RunPod serverless handler
    import runpod

    log.info("Starting RunPod queue handler")
    log.info(f"  embed_model={config.embed_model or 'disabled'}")
    log.info(f"  rerank_model={config.rerank_model or 'disabled'}")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda _: config.max_concurrency,
    })
