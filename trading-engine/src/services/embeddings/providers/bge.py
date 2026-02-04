"""
BGE-small-en embedding provider (free, local).

Uses sentence-transformers to run BGE-small-en locally.
384-dimensional embeddings optimized for semantic search.

Requires optional dependency: uv sync --extra embedding
"""

import asyncio
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class BGESmallProvider:
    """
    BGE-small-en embedding provider (free, local).

    Uses BAAI/bge-small-en-v1.5 model via sentence-transformers.
    Produces 384-dimensional embeddings optimized for semantic similarity search.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize BGE-small-en provider.

        Args:
            model_name: Model name (defaults to config value)
        """
        self.model_name = model_name or settings.embedding.model_name
        self._model: SentenceTransformer | None = None
        self._dimensions = 384  # BGE-small-en fixed dimensions

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model (loads on first use)."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. Install with: uv sync --extra embedding"
            )
        if self._model is None:
            log.info("loading_bge_model", model_name=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log.info("bge_model_loaded", model_name=self.model_name)
        return self._model

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector
        """
        # Run in thread pool to avoid blocking event loop
        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False,
        )
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient than individual calls).

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embedding vectors
        """
        # Run in thread pool to avoid blocking event loop
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

