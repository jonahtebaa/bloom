"""Voyage AI embeddings adapter — Anthropic's recommended embedding provider.

Anthropic does not currently ship a first-party embeddings API, so we use
Voyage AI (voyageai.com) as the canonical pairing. Requires `pip install
bloom-mcp[anthropic]` and a `VOYAGE_API_KEY` env var.

Voyage's SDK has no per-call timeout knob, so we rely on the recall/remember
caller to wrap embedder errors and fall back to keyword scoring on failure.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class VoyageEmbedder:
    name = "voyage"

    def __init__(self, model: str = "voyage-3-lite") -> None:
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "Voyage embedder requires the `anthropic` extra. "
                "Install with: pip install bloom-mcp[anthropic]"
            ) from e

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set. Get one at https://www.voyageai.com/ "
                "or run `bloom-mcp init` to configure your embedder."
            )

        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dim = 512 if "lite" in model else 1024

    def _embed(self, text: str, input_type: str) -> "np.ndarray":
        import numpy as np

        result = self._client.embed([text], model=self.model, input_type=input_type)
        return np.asarray(result.embeddings[0], dtype=np.float32)

    def embed_doc(self, content: str) -> "np.ndarray":
        return self._embed(content, input_type="document")

    def embed_query(self, query: str) -> "np.ndarray":
        return self._embed(query, input_type="query")

    def embed_batch(self, texts: list[str]) -> list["np.ndarray"]:
        import numpy as np

        if not texts:
            return []
        result = self._client.embed(texts, model=self.model, input_type="document")
        return [np.asarray(v, dtype=np.float32) for v in result.embeddings]
