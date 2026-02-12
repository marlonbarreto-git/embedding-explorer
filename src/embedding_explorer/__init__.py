"""Embedding Explorer - Visualize and compare text embeddings across models."""

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "Point2D",
    "SimilarityPair",
    "Visualizer",
]

__version__ = "0.1.0"

from .embedder import Embedder, EmbeddingResult
from .visualizer import Point2D, SimilarityPair, Visualizer
