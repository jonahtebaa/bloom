"""Embedder protocol — every provider implements this."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Anything that turns text into a fixed-length float vector."""

    name: str
    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for `text`."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""
        ...
