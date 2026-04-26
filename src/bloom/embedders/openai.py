"""OpenAI embeddings adapter.

Requires `pip install bloom-mcp[openai]` and an `OPENAI_API_KEY` env var.
The HTTP client is created once and reused; every embedding call is bounded
by a 10s timeout so a flaky network never stalls `recall` or `remember` —
recall callers wrap embedder errors in try/except and fall back to keyword
scoring on failure.

Dim discovery: we hint at the dim from a known-models table for fast
construction, but the FIRST embed call confirms `self.dim` from the actual
returned vector and persists it to `~/.bloom/.embedder_cache.json` keyed
by `(provider, model)`. This means custom or newly-released models work
out of the box rather than landing on the wrong default and silently
breaking cosine math.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from bloom.embedders._dim_cache import get_cached_dim, set_cached_dim

if TYPE_CHECKING:
    import numpy as np


_DEFAULT_TIMEOUT_S = 10.0

# Known model → dim. Used as a startup hint; the real dim is confirmed
# from the first probe and persisted to the cache.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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

        # Prefer cache, then known-model table, then 0 (probe-on-first-call).
        # `_dim_confirmed` flips True only after we've seen a real vector
        # so a wrong cached value or stale known-models entry corrects itself.
        cached = get_cached_dim(self.name, model)
        if cached is not None:
            self.dim = cached
            self._dim_confirmed = True
        else:
            self.dim = _KNOWN_DIMS.get(model, 0)
            self._dim_confirmed = False

    def _ensure_dim(self, vec: "np.ndarray") -> None:
        """Confirm self.dim from the first real embedding response."""
        if self._dim_confirmed:
            return
        observed = int(vec.size)
        if observed <= 0:
            return
        if self.dim and observed != self.dim:
            print(
                f"[bloom] openai embedder: model {self.model!r} returned dim "
                f"{observed} (expected {self.dim} from defaults) — using "
                f"observed value.",
                file=sys.stderr,
            )
        self.dim = observed
        self._dim_confirmed = True
        set_cached_dim(self.name, self.model, observed)

    def _embed_one(self, text: str) -> "np.ndarray":
        import numpy as np

        resp = self._client.embeddings.create(model=self.model, input=text)
        vec = resp.data[0].embedding
        arr = np.asarray(vec, dtype=np.float32)
        self._ensure_dim(arr)
        return arr

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
        out = [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
        if out:
            self._ensure_dim(out[0])
        return out
