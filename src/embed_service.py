import time
import numpy as np
from config import Config
from schemas import embedding_response
from logger import log


class EmbedService:
    """SentenceTransformer wrapper with lazy model loading."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    async def ensure_loaded(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"dtype": self.config.dtype, "device_map": "auto"}
        attn = "sdpa"
        if self.config.use_flash_attn:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                attn = "flash_attention_2"
        # Explicitly set attn implementation to prevent transformers from
        # auto-detecting a potentially broken flash-attn install
        model_kwargs["attn_implementation"] = attn

        log.info(f"Loading embed model: {self.config.embed_model}")
        log.info(f"  attn={attn}, dtype={self.config.dtype}, batch_size={self.config.embed_batch_size}")
        log.info(f"  max_context_length={self.config.max_context_length or 'model default'}, max_client_batch_size={self.config.max_client_batch_size or 'unlimited'}")
        log.info(f"  default_dimensions={self.config.default_dimensions or 'native (no truncation)'}")
        start = time.perf_counter()
        self.model = SentenceTransformer(
            self.config.embed_model,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        if self.config.max_context_length:
            self.model.max_seq_length = self.config.max_context_length
        elapsed = time.perf_counter() - start
        log.info(f"Embed model loaded in {elapsed:.1f}s — device={self.model.device}, max_seq={self.model.max_seq_length}")

    def embed(self, payload):
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        if self.config.max_client_batch_size and len(texts) > self.config.max_client_batch_size:
            return {"error": f"Batch size {len(texts)} exceeds MAX_CLIENT_BATCH_SIZE ({self.config.max_client_batch_size})"}

        dimensions = payload.get("dimensions") or self.config.default_dimensions

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # MRL truncation: slice to requested dimensions and re-normalize
        if dimensions and dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :dimensions]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        # Token count
        encoded = self.model.tokenizer(texts, padding=False, truncation=False)
        total_tokens = sum(len(ids) for ids in encoded["input_ids"])

        return embedding_response(
            model_name=self.config.embed_model,
            embeddings=[emb.tolist() for emb in embeddings],
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )
