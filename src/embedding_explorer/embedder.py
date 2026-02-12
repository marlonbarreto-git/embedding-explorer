"""Embedding generation using sentence-transformers."""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {"dim": 384, "description": "Fast, good quality (default)"},
    "all-mpnet-base-v2": {"dim": 768, "description": "Best quality, slower"},
    "paraphrase-MiniLM-L3-v2": {"dim": 384, "description": "Fastest, lower quality"},
}


@dataclass
class EmbeddingResult:
    texts: list[str]
    embeddings: np.ndarray
    model_name: str
    dimension: int

    @property
    def count(self) -> int:
        return len(self.texts)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
            )
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return AVAILABLE_MODELS[self._model_name]["dim"]

    def embed(self, texts: list[str]) -> EmbeddingResult:
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
