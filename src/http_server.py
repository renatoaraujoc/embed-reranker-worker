import time
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from logger import log


def create_app(config, embed_service, rerank_service):
    app = FastAPI()

    @app.on_event("startup")
    async def startup():
        log.info("Loading models at startup (load balancer mode)...")
        if embed_service:
            await embed_service.ensure_loaded()
        if rerank_service:
            await rerank_service.ensure_loaded()
        log.info("Models ready — /ping will return 200")

    @app.get("/ping")
    async def ping():
        # 200 = ready, 204 = still initializing
        loading = (embed_service and embed_service.model is None) or \
                  (rerank_service and rerank_service.model is None)
        if loading:
            return Response(status_code=204)
        return JSONResponse(status_code=200, content={"status": "healthy"})

    @app.get("/v1/models")
    async def models():
        data = []
        if config.embed_model:
            data.append({"id": config.embed_model, "object": "model", "type": "embedding"})
        if config.rerank_model:
            data.append({"id": config.rerank_model, "object": "model", "type": "reranker"})
        return {"object": "list", "data": data}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if not embed_service:
            return JSONResponse(status_code=400, content={"error": "Embeddings not configured (EMBED_MODEL not set)"})
        payload = await request.json()
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        log.info(f"POST /v1/embeddings — {len(texts)} texts, dims={payload.get('dimensions', 'native')}")
        start = time.perf_counter()
        await embed_service.ensure_loaded()
        result = embed_service.embed(payload)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(f"Embeddings complete — {elapsed:.0f}ms")
        return result

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        if not rerank_service:
            return JSONResponse(status_code=400, content={"error": "Reranking not configured (RERANK_MODEL not set)"})
        payload = await request.json()
        log.info(f"POST /v1/rerank — {len(payload.get('documents', []))} docs, top_n={payload.get('top_n', 'all')}")
        start = time.perf_counter()
        await rerank_service.ensure_loaded()
        result = rerank_service.rerank(payload)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(f"Rerank complete — {elapsed:.0f}ms")
        return result

    return app
