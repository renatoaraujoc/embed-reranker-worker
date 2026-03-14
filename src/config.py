import os


class Config:
    """Configuration from environment variables. No model loading."""

    def __init__(self):
        self.embed_model = os.environ.get("EMBED_MODEL")
        self.rerank_model = os.environ.get("RERANK_MODEL")
        if not self.embed_model and not self.rerank_model:
            raise ValueError("At least one of EMBED_MODEL or RERANK_MODEL must be set")
        self.embed_batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
        self.rerank_batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "32"))
        self.dtype = os.environ.get("DTYPE", "auto")
        self.use_flash_attn = os.environ.get("USE_FLASH_ATTN", "true").lower() == "true"
        self.default_dimensions = int(os.environ.get("EMBED_DEFAULT_DIMENSIONS", "0")) or None
        self.max_context_length = int(os.environ.get("MAX_CONTEXT_LENGTH", "0")) or None
        self.max_client_batch_size = int(os.environ.get("MAX_CLIENT_BATCH_SIZE", "0")) or None
        self.max_concurrency = int(os.environ.get("RUNPOD_MAX_CONCURRENCY", "1"))
