"""OpenAI embeddings adapter.

Requires `pip install bloom-mcp[openai]` and an `OPENAI_API_KEY` env var.
The HTTP client is created once and reused; every embedding call is bounded
by a 10s timeout so a flaky network never stalls `recall` or `remember` —
recall callers wrap embedder errors in try/except and fall back to keyword
scoring on failure.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


_DEFAULT_TIMEOUT_S = 10.0


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI embedder requires the `openai` extra. "
                "Install with: pip install bloom-mcp[openai]"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or run `bloom-mcp init` "
                "to configure your embedder."
            )

        # Cache the client — instantiating per-call would force a new TLS
        # handshake on every embed_doc/embed_query.
        self._client = OpenAI(api_key=api_key, timeout=_DEFAULT_TIMEOUT_S)
        self.model = model
        # text-embedding-3-small = 1536, text-embedding-3-large = 3072.
        self.dim = 1536 if "small" in model else 3072

    def _embed_one(self, text: str) -> "np.ndarray":
        import numpy as np

        resp = self._client.embeddings.create(model=self.model, input=text)
        vec = resp.data[0].embedding
        return np.asarray(vec, dtype=np.float32)

    def embed_doc(self, content: str) -> "np.ndarray":
        return self._embed_one(content)

    def embed_query(self, query: str) -> "np.ndarray":
        return self._embed_one(query)

    def embed_batch(self, texts: list[str]) -> list["np.ndarray"]:
        """Batched variant used by the backfill command."""
        import numpy as np

        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
