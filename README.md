# Embedding Explorer

Visual tool for understanding text embeddings with 2D projection and similarity analysis.

## Overview

Embedding Explorer generates embeddings using sentence-transformers, reduces them to 2D with PCA for visualization, and computes pairwise cosine similarity to find the most related text pairs. It supports multiple embedding models so you can compare quality and dimensionality trade-offs.

## Architecture

```
Input Texts
  |
  v
Embedder (sentence-transformers)
  |
  +---> all-MiniLM-L6-v2       (384d, fast)
  +---> all-mpnet-base-v2      (768d, best quality)
  +---> paraphrase-MiniLM-L3-v2 (384d, fastest)
  |
  v
EmbeddingResult (texts + numpy array + dimension)
  |
  v
Visualizer
  |
  +---> PCA reduction       -> list[Point2D]
  +---> Similarity matrix   -> numpy NxN array
  +---> Top-k similar pairs -> list[SimilarityPair]
```

## Features

- Generate embeddings with multiple sentence-transformer models
- Dimensionality reduction to 2D via PCA for visualization
- Full cosine similarity matrix computation
- Find top-k most similar text pairs from a corpus
- Three pre-configured models with different quality/speed trade-offs

## Tech Stack

- Python 3.11+
- sentence-transformers
- NumPy
- scikit-learn (PCA, cosine similarity)
- FastAPI + Uvicorn
- Pydantic

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/embedding-explorer.git
cd embedding-explorer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/embedding_explorer/
  __init__.py
  embedder.py      # Embedding generation with sentence-transformers
  visualizer.py    # PCA reduction, similarity matrix, top-k pairs
tests/
  test_embedder.py
  test_visualizer.py
```

## Testing

```bash
pytest -v --cov=src/embedding_explorer
```

18 tests covering embedding generation, model selection, PCA reduction, and similarity analysis.

## License

MIT
