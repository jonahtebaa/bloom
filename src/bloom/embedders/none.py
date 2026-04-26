"""No-op embedder. Returns empty vectors; recall falls back to keyword scoring."""

from __future__ import annotations


class NoneEmbedder:
    name = "none"
    dim = 0

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
