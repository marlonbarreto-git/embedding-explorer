"""Tests for the embedding generation module."""

import numpy as np
import pytest

from embedding_explorer.embedder import AVAILABLE_MODELS, Embedder, EmbeddingResult


class TestEmbedder:
    def test_create_default(self):
        embedder = Embedder()
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.dimension == 384

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Embedder(model_name="nonexistent-model")

    def test_embed_texts(self):
        embedder = Embedder()
        result = embedder.embed(["Hello world", "Goodbye world"])
        assert isinstance(result, EmbeddingResult)
        assert result.count == 2
        assert result.embeddings.shape == (2, 384)
        assert result.model_name == "all-MiniLM-L6-v2"

    def test_embed_single_text(self):
        embedder = Embedder()
        result = embedder.embed(["Single text"])
        assert result.count == 1
        assert result.embeddings.shape[0] == 1

    def test_embed_empty_list(self):
        embedder = Embedder()
        result = embedder.embed([])
        assert result.count == 0

    def test_embeddings_are_normalized(self):
        embedder = Embedder()
        result = embedder.embed(["Test sentence"])
        norm = np.linalg.norm(result.embeddings[0])
        assert abs(norm - 1.0) < 0.01  # sentence-transformers normalizes by default

    def test_similar_texts_have_high_similarity(self):
        embedder = Embedder()
        result = embedder.embed(["I love programming", "I enjoy coding"])
        sim = np.dot(result.embeddings[0], result.embeddings[1])
        assert sim > 0.5  # Similar sentences should have high cosine similarity

    def test_different_texts_have_lower_similarity(self):
        embedder = Embedder()
        result = embedder.embed(["I love programming", "The weather is sunny today"])
        sim = np.dot(result.embeddings[0], result.embeddings[1])
        assert sim < 0.5


class TestAvailableModels:
    def test_has_expected_models(self):
        assert "all-MiniLM-L6-v2" in AVAILABLE_MODELS
        assert "all-mpnet-base-v2" in AVAILABLE_MODELS

    def test_model_info(self):
        info = AVAILABLE_MODELS["all-MiniLM-L6-v2"]
        assert "dim" in info
        assert "description" in info
