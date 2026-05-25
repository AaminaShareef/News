"""
processing/embedder.py
----------------------
Stage 2 of the ML pipeline.
Converts clean_text into 384-dim vectors using all-MiniLM-L6-v2.
Exactly as specified in doc section 5.1.

Model: sentence-transformers/all-MiniLM-L6-v2
Speed: ~40ms per article
Output: 384-number embedding vector per article
"""

import logging
import numpy as np
from typing import List

log = logging.getLogger(__name__)

_model = None  # Singleton — loaded once, reused across batches


def _load_model():
    global _model
    if _model is None:
        log.info("Loading DistilBERT model: sentence-transformers/all-MiniLM-L6-v2 ...")
        try:
            from sentence_transformers import SentenceTransformer
            from config.settings import DISTILBERT_MODEL
            _model = SentenceTransformer(DISTILBERT_MODEL)
            log.info("Model loaded. (~80MB, ~40ms per article)")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_articles(articles: List[dict], batch_size: int = 64) -> List[dict]:
    """
    Embed each article's clean_text into a 384-dim vector.
    Adds 'embedding' key (list of floats) to each article dict.

    Doc spec: title + first 300 chars of body → 384-number vector.
    The clean_text field from cleaner.py already contains this.

    Args:
        articles:   list of cleaned article dicts (must have 'clean_text')
        batch_size: articles per forward pass (tune for GPU/CPU memory)

    Returns:
        Same list with 'embedding' added to each dict.
    """
    model = _load_model()

    texts = [a.get("clean_text", "") for a in articles]

    log.info("Embedding %d articles (batch_size=%d) ...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity = dot product after normalise
    )

    for article, emb in zip(articles, embeddings):
        article["embedding"] = emb.tolist()

    log.info("Embedding complete. Vector dim: %d", embeddings.shape[1])
    return articles


def embed_text(text: str) -> np.ndarray:
    """Embed a single string. Used for query-time search."""
    model = _load_model()
    return model.encode([text], normalize_embeddings=True)[0]