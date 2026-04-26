"""Voyage AI embeddings adapter — Anthropic's recommended embedding provider.

Anthropic does not currently ship a first-party embeddings API, so we use
Voyage AI (voyageai.com) as the canonical pairing. Requires `pip install
bloom-mcp[anthropic]` and a VOYAGE_API_KEY env var.
"""

from __future__ import annotations

import os


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

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.model)
        return list(result.embeddings[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self.model)
        return [list(v) for v in result.embeddings]
