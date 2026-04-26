"""Voyage AI embeddings adapter — Anthropic's recommended embedding provider.

Anthropic does not currently ship a first-party embeddings API, so we use
Voyage AI (voyageai.com) as the canonical pairing. Requires `pip install
bloom-mcp[anthropic]` and a `VOYAGE_API_KEY` env var.

Voyage's SDK has no per-call timeout knob, so we rely on the recall/remember
caller to wrap embedder errors and fall back to keyword scoring on failure.

Dim discovery: like OpenAIEmbedder, we hint from a known-models table at
construction time but confirm `self.dim` from the first real probe and
persist it to `~/.bloom/.embedder_cache.json`. New / custom Voyage models
work without code changes.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from bloom.embedders._dim_cache import get_cached_dim, set_cached_dim

if TYPE_CHECKING:
    import numpy as np


# Known model → dim. Used as a startup hint; the real dim is confirmed from
# the first probe and persisted to the cache.
_KNOWN_DIMS = {
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-multilingual-2": 1024,
}


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

        cached = get_cached_dim(self.name, model)
        if cached is not None:
            self.dim = cached
            self._dim_confirmed = True
        else:
            self.dim = _KNOWN_DIMS.get(model, 0)
            self._dim_confirmed = False

    def _ensure_dim(self, vec: "np.ndarray") -> None:
        if self._dim_confirmed:
            return
        observed = int(vec.size)
        if observed <= 0:
            return
        if self.dim and observed != self.dim:
            print(
                f"[bloom] voyage embedder: model {self.model!r} returned dim "
                f"{observed} (expected {self.dim} from defaults) — using "
                f"observed value.",
                file=sys.stderr,
            )
        self.dim = observed
        self._dim_confirmed = True
        set_cached_dim(self.name, self.model, observed)

    def _embed(self, text: str, input_type: str) -> "np.ndarray":
        import numpy as np

        result = self._client.embed([text], model=self.model, input_type=input_type)
        arr = np.asarray(result.embeddings[0], dtype=np.float32)
        self._ensure_dim(arr)
        return arr

    def embed_doc(self, content: str) -> "np.ndarray":
        return self._embed(content, input_type="document")

    def embed_query(self, query: str) -> "np.ndarray":
        return self._embed(query, input_type="query")

    def embed_batch(self, texts: list[str]) -> list["np.ndarray"]:
        import numpy as np

        if not texts:
            return []
        result = self._client.embed(texts, model=self.model, input_type="document")
        out = [np.asarray(v, dtype=np.float32) for v in result.embeddings]
        if out:
            self._ensure_dim(out[0])
        return out
