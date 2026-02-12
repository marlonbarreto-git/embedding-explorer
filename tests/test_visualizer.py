"""Tests for the visualization module."""

import numpy as np
import pytest

from embedding_explorer.embedder import Embedder, EmbeddingResult
from embedding_explorer.visualizer import Point2D, SimilarityPair, Visualizer


class TestVisualizer:
    def setup_method(self):
        self.visualizer = Visualizer()
        self.embedder = Embedder()

    def test_reduce_to_2d(self):
        result = self.embedder.embed([
            "Python programming",
            "JavaScript development",
            "Machine learning",
            "Deep learning",
            "Web development",
        ])
        points = self.visualizer.reduce_to_2d(result)
        assert len(points) == 5
        assert all(isinstance(p, Point2D) for p in points)
        assert all(isinstance(p.x, float) for p in points)

    def test_reduce_empty(self):
        result = self.embedder.embed([])
        points = self.visualizer.reduce_to_2d(result)
        assert points == []

    def test_reduce_few_texts(self):
        result = self.embedder.embed(["Hello", "World"])
        points = self.visualizer.reduce_to_2d(result)
        assert len(points) == 2

    def test_similarity_matrix(self):
        result = self.embedder.embed(["A", "B", "C"])
        matrix = self.visualizer.compute_similarity_matrix(result)
        assert matrix.shape == (3, 3)
        # Diagonal should be ~1.0 (self-similarity)
        for i in range(3):
            assert abs(matrix[i, i] - 1.0) < 0.01

    def test_similarity_matrix_empty(self):
        result = self.embedder.embed([])
        matrix = self.visualizer.compute_similarity_matrix(result)
        assert len(matrix) == 0

    def test_find_most_similar(self):
        result = self.embedder.embed([
            "I love dogs",
            "I adore puppies",
            "The stock market crashed",
            "Financial markets declined",
        ])
        pairs = self.visualizer.find_most_similar(result, top_k=2)
        assert len(pairs) == 2
        assert all(isinstance(p, SimilarityPair) for p in pairs)
        # Most similar pair should have highest score
        assert pairs[0].score >= pairs[1].score

    def test_find_most_similar_single_text(self):
        result = self.embedder.embed(["Only one text"])
        pairs = self.visualizer.find_most_similar(result)
        assert pairs == []

    def test_points_have_text(self):
        texts = ["Alpha", "Beta", "Gamma", "Delta"]
        result = self.embedder.embed(texts)
        points = self.visualizer.reduce_to_2d(result)
        point_texts = {p.text for p in points}
        assert point_texts == set(texts)
