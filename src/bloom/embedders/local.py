"""Local embeddings via sentence-transformers — fully offline, no API key.

Requires `pip install bloom-mcp[local]`. First run downloads model weights
(~80 MB for the default MiniLM).
"""

from __future__ import annotations


class LocalEmbedder:
    name = "local"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embedder requires the `local` extra. "
                "Install with: pip install bloom-mcp[local]"
            ) from e

        self._model = SentenceTransformer(model)
        self.model = model
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=False)
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=False)
        return [[float(x) for x in v] for v in vecs]
