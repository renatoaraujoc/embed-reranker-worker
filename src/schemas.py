def embedding_response(model_name, embeddings, usage):
    """Build OpenAI-compatible /v1/embeddings response."""
    return {
        "object": "list",
        "model": model_name,
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "usage": usage,
    }


def rerank_response(results, usage):
    """Build /v1/rerank response."""
    return {
        "results": results,
        "usage": usage,
    }


def models_response(models):
    """Build /v1/models response."""
    return {
        "object": "list",
        "data": models,
    }
