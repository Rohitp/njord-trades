"""
Embedding service for generating and managing vector embeddings.

Orchestrates embedding generation using configured provider (BGE-small-en by default).
"""

from typing import List

from src.config import settings
from src.services.embeddings.providers.bge import BGESmallProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating vector embeddings.

    Supports multiple providers (BGE-small-en, OpenAI, etc.) with BGE-small-en as default.
    """

    def __init__(self, provider: str | None = None):
        """
        Initialize embedding service.

        Args:
            provider: Provider name (defaults to config value: "bge-small")
            
        Raises:
            ImportError: If embedding dependencies are not installed
        """
        provider = provider or settings.embedding.provider
        self.provider_name = provider

        try:
            if provider == "bge-small":
                self.provider = BGESmallProvider()
            else:
                raise ValueError(f"Unknown embedding provider: {provider}")

            self.dimensions = self.provider.get_dimensions()
            log.info("embedding_service_initialized", provider=provider, dimensions=self.dimensions)
        except ImportError as e:
            error_msg = "sentence-transformers not installed. Install with: uv sync --extra embedding"
            log.error("embedding_service_init_failed", error=error_msg, provider=provider)
            raise ImportError(error_msg) from e

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (384 dimensions for BGE-small-en)
            
        Raises:
            ImportError: If embedding dependencies are not installed
        """
        try:
            return await self.provider.embed(text)
        except ImportError:
            error_msg = "sentence-transformers not installed. Install with: uv sync --extra embedding"
            log.error("embedding_failed", error=error_msg)
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self.provider.embed_batch(texts)

    def get_dimensions(self) -> int:
        """Get embedding dimensions for this provider."""
        return self.dimensions

