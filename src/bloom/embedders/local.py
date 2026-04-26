"""Local embeddings via sentence-transformers — fully offline, no API key.

Requires `pip install bloom-mcp[local]`. First run downloads model weights
(~80 MB for the default MiniLM); we print a one-time stderr notice so the
user isn't surprised by an opaque pause on first use.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


_NOTICE_PRINTED = False


def _notice_first_use(model_name: str) -> None:
    global _NOTICE_PRINTED
    if _NOTICE_PRINTED:
        return
    _NOTICE_PRINTED = True
    print(
        f"[bloom] loading local embedder ({model_name}) — "
        "first run may download ~80 MB of weights",
        file=sys.stderr,
    )


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

        _notice_first_use(model)
        self._model = SentenceTransformer(model)
        self.model = model
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)

    def _encode(self, text: str) -> "np.ndarray":
        import numpy as np

        vec = self._model.encode(text, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_doc(self, content: str) -> "np.ndarray":
        return self._encode(content)

    def embed_query(self, query: str) -> "np.ndarray":
        return self._encode(query)

    def embed_batch(self, texts: list[str]) -> list["np.ndarray"]:
        import numpy as np

        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [np.asarray(v, dtype=np.float32) for v in vecs]
