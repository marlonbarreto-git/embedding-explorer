# embedding-explorer

Visual tool for understanding text embeddings. Generate embeddings with sentence-transformers, visualize in 2D with PCA, compute similarity matrices, and find the most similar text pairs.

## Features

- **Multi-model support**: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L3-v2
- **2D visualization**: PCA-based dimensionality reduction for plotting
- **Similarity analysis**: Cosine similarity matrix and top-k most similar pairs
- **Normalized embeddings**: All vectors are unit-normalized for accurate cosine similarity

## Architecture

```
embedding_explorer/
├── embedder.py     # Embedding generation with sentence-transformers
└── visualizer.py   # PCA reduction, similarity matrix, pair ranking
```

## Quick Start

```python
from embedding_explorer.embedder import Embedder
from embedding_explorer.visualizer import Visualizer

embedder = Embedder(model_name="all-MiniLM-L6-v2")
result = embedder.embed([
    "Python is great for AI",
    "Machine learning with Python",
    "JavaScript for web dev",
    "React frontend development",
])

viz = Visualizer()
points = viz.reduce_to_2d(result)       # 2D coordinates for plotting
pairs = viz.find_most_similar(result)    # Most similar text pairs
matrix = viz.compute_similarity_matrix(result)  # Full NxN similarity
```

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v
```

## Roadmap

- **v2**: Compare multiple embedding models side-by-side, interactive semantic search
- **v3**: Auto clustering, model recommendation per dataset, export results

## License

MIT
