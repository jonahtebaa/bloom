"""Embedder protocol — every provider implements this.

The protocol covers two surfaces:
  * `embed_doc(content)` — write-time, called once per `remember`.
  * `embed_query(query)` — read-time, called once per `recall`.

Both return a numpy float32 vector of length `dim`. The no-op `none` embedder
returns an empty array and exposes `dim = 0`; callers must guard with
`embedder.dim > 0` before doing similarity math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class Embedder(Protocol):
    """Anything that turns text into a fixed-length float32 vector."""

    name: str
    dim: int

    def embed_doc(self, content: str) -> "np.ndarray":
        """Embed a stored document (write path)."""
        ...

    def embed_query(self, query: str) -> "np.ndarray":
        """Embed a search query (read path)."""
        ...
