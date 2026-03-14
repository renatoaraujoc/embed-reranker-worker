import time
import torch
from config import Config
from schemas import rerank_response
from logger import log


class RerankService:
    """CrossEncoder wrapper with lazy model loading."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    async def ensure_loaded(self):
        if self.model is not None:
            return
        from sentence_transformers import CrossEncoder

        kwargs = {
            "trust_remote_code": True,
            "model_kwargs": {"dtype": self.config.dtype},
        }
        # Explicitly target CUDA on GPU machines; otherwise let
        # sentence-transformers auto-detect (MPS on Mac, CPU fallback).
        if torch.cuda.is_available():
            kwargs["device"] = "cuda"

        log.info(f"Loading rerank model: {self.config.rerank_model}")
        log.info(f"  dtype={self.config.dtype}, batch_size={self.config.rerank_batch_size}")
        log.info(f"  max_context_length={self.config.max_context_length or 'model default'}, max_client_batch_size={self.config.max_client_batch_size or 'unlimited'}")
        start = time.perf_counter()
        self.model = CrossEncoder(self.config.rerank_model, **kwargs)
        if self.config.max_context_length:
            self.model.max_length = self.config.max_context_length
        elapsed = time.perf_counter() - start
        log.info(f"Rerank model loaded in {elapsed:.1f}s — device={self.model.device}, max_seq={self.model.max_length}")

    def rerank(self, payload):
        query = payload["query"]
        documents = payload["documents"]

        if self.config.max_client_batch_size and len(documents) > self.config.max_client_batch_size:
            return {"error": f"Batch size {len(documents)} exceeds MAX_CLIENT_BATCH_SIZE ({self.config.max_client_batch_size})"}
        top_n = payload.get("top_n", len(documents))
        return_documents = payload.get("return_documents", True)

        results = self.model.rank(
            query=query,
            documents=documents,
            top_k=top_n,
            return_documents=return_documents,
            batch_size=self.config.rerank_batch_size,
        )

        # Token count for all query-document pairs
        encoded = self.model.tokenizer(
            [query] * len(documents),
            documents,
            padding=False,
            truncation=False,
        )
        total_tokens = sum(len(ids) for ids in encoded["input_ids"])

        # Map sentence-transformers format → API response format
        mapped = []
        for r in results:
            entry = {
                "index": r["corpus_id"],
                "relevance_score": float(r["score"]),
            }
            if return_documents:
                entry["document"] = {"text": r["text"]}
            mapped.append(entry)

        return rerank_response(
            results=mapped,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )
