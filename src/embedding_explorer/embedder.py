"""Embedding generation using sentence-transformers."""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {"dim": 384, "description": "Fast, good quality (default)"},
    "all-mpnet-base-v2": {"dim": 768, "description": "Best quality, slower"},
    "paraphrase-MiniLM-L3-v2": {"dim": 384, "description": "Fastest, lower quality"},
}


@dataclass
class EmbeddingResult:
    """Container for embedding results with associated metadata."""

    texts: list[str]
    embeddings: np.ndarray
    model_name: str
    dimension: int

    @property
    def count(self) -> int:
        """Return the number of embedded texts."""
        return len(self.texts)


class Embedder:
    """Generates text embeddings using sentence-transformer models."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize the embedder with a sentence-transformer model.

        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
            )
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the loaded model."""
        return AVAILABLE_MODELS[self._model_name]["dim"]

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: Strings to embed.

        Returns:
            An EmbeddingResult containing the texts and their embeddings.
        """
        if not texts:
            return EmbeddingResult(
                texts=[], embeddings=np.array([]), model_name=self._model_name, dimension=self.dimension
            )

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return EmbeddingResult(
            texts=texts,
            embeddings=embeddings,
            model_name=self._model_name,
            dimension=embeddings.shape[1],
        )
