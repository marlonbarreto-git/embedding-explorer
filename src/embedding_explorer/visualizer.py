"""Dimensionality reduction and visualization of embeddings."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from embedding_explorer.embedder import EmbeddingResult


@dataclass
class Point2D:
    x: float
    y: float
    text: str
    cluster: int = -1


@dataclass
class SimilarityPair:
    text_a: str
    text_b: str
    score: float


class Visualizer:
    def reduce_to_2d(self, result: EmbeddingResult) -> list[Point2D]:
        if result.count == 0:
            return []

        if result.count < 3:
            coords = result.embeddings[:, :2]
        else:
            n_components = min(2, result.count, result.embeddings.shape[1])
            reducer = PCA(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(result.embeddings)

        return [
            Point2D(x=float(coords[i, 0]), y=float(coords[i, 1]), text=result.texts[i])
            for i in range(len(result.texts))
        ]

    def compute_similarity_matrix(self, result: EmbeddingResult) -> np.ndarray:
        if result.count == 0:
            return np.array([])
        return cosine_similarity(result.embeddings)

    def find_most_similar(
        self, result: EmbeddingResult, top_k: int = 5
    ) -> list[SimilarityPair]:
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
