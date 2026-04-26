"""Pluggable embedder providers.

The default `none` embedder makes Bloom work offline with zero dependencies.
Other providers (openai, anthropic-via-voyage, local sentence-transformers)
are opt-in extras — install with e.g. `pip install bloom-mcp[openai]`.

`load_embedder` is the single entry point. It is forgiving by design: if the
configured provider isn't available (missing optional package, missing API
key, unknown name), we log a clear warning to stderr and fall back to the
no-op `NoneEmbedder` so the server keeps working with keyword recall.
"""

from __future__ import annotations

import sys

from bloom.config import EmbedderConfig
from bloom.embedders.base import Embedder
from bloom.embedders.none import NoneEmbedder


def _warn(msg: str) -> None:
    print(f"[bloom] embedder: {msg} — falling back to keyword-only recall", file=sys.stderr)


def load_embedder(cfg: EmbedderConfig) -> Embedder:
    """Resolve the configured embedder, importing optional deps lazily.

    On any error (unknown provider, ImportError, missing API key, init
    crash) returns a NoneEmbedder rather than raising — so a misconfigured
    embedder never takes the whole server down.
    """
    provider = (cfg.provider or "none").lower()
    if provider == "none":
        return NoneEmbedder()

    try:
        if provider == "openai":
            from bloom.embedders.openai import OpenAIEmbedder

            return OpenAIEmbedder(model=cfg.model or "text-embedding-3-small")
        if provider in ("anthropic", "voyage"):
            from bloom.embedders.anthropic import VoyageEmbedder

            return VoyageEmbedder(model=cfg.model or "voyage-3-lite")
        if provider == "local":
            from bloom.embedders.local import LocalEmbedder

            return LocalEmbedder(model=cfg.model or "all-MiniLM-L6-v2")
    except ImportError as e:
        _warn(f"{provider}: {e}")
        return NoneEmbedder()
    except Exception as e:  # noqa: BLE001 — missing API key, network, etc.
        _warn(f"{provider}: {e}")
        return NoneEmbedder()

    _warn(f"unknown provider {provider!r}")
    return NoneEmbedder()


__all__ = ["Embedder", "NoneEmbedder", "load_embedder"]
