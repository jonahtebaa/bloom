"""OpenAI embeddings adapter.

Requires `pip install bloom-mcp[openai]` and an OPENAI_API_KEY env var.
"""

from __future__ import annotations

import os


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

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = 1536 if "small" in model else 3072

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in resp.data]
