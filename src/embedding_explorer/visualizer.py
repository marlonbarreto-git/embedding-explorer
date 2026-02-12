"""Dimensionality reduction and visualization of embeddings."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from embedding_explorer.embedder import EmbeddingResult

TARGET_DIMENSIONS = 2
MIN_POINTS_FOR_PCA = 3
PCA_RANDOM_STATE = 42
DEFAULT_TOP_K = 5
DEFAULT_CLUSTER = -1  # Indicates no cluster assignment


@dataclass
class Point2D:
    """A 2D point representing a projected embedding with optional cluster label."""

    x: float
    y: float
    text: str
    cluster: int = DEFAULT_CLUSTER


@dataclass
class SimilarityPair:
    """A pair of texts with their cosine similarity score."""

    text_a: str
    text_b: str
    score: float


class Visualizer:
    """Reduces embedding dimensions and computes text similarity."""

    def reduce_to_2d(self, result: EmbeddingResult) -> list[Point2D]:
        """Project high-dimensional embeddings to 2D using PCA.

        Args:
            result: Embedding result to reduce.

        Returns:
            List of 2D points, one per input text.
        """
        if result.count == 0:
            return []

        if result.count < MIN_POINTS_FOR_PCA:
            coords = result.embeddings[:, :TARGET_DIMENSIONS]
        else:
            n_components = min(TARGET_DIMENSIONS, result.count, result.embeddings.shape[1])
            reducer = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            coords = reducer.fit_transform(result.embeddings)

        return [
            Point2D(x=float(coords[i, 0]), y=float(coords[i, 1]), text=result.texts[i])
            for i in range(len(result.texts))
        ]

    def compute_similarity_matrix(self, result: EmbeddingResult) -> np.ndarray:
        """Compute pairwise cosine similarity for all embeddings.

        Args:
            result: Embedding result to compare.

        Returns:
            Symmetric similarity matrix as a numpy array.
        """
        if result.count == 0:
            return np.array([])
        return cosine_similarity(result.embeddings)

    def find_most_similar(
        self, result: EmbeddingResult, top_k: int = DEFAULT_TOP_K
    ) -> list[SimilarityPair]:
        """Find the most similar text pairs by cosine similarity.

        Args:
            result: Embedding result to search.
            top_k: Maximum number of pairs to return.

        Returns:
            List of similarity pairs sorted by descending score.
        """
        if result.count < 2:
            return []

        sim_matrix = self.compute_similarity_matrix(result)
        pairs = []

        for i in range(len(result.texts)):
            for j in range(i + 1, len(result.texts)):
                pairs.append(
                    SimilarityPair(
                        text_a=result.texts[i],
                        text_b=result.texts[j],
                        score=float(sim_matrix[i, j]),
                    )
                )

        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs[:top_k]
