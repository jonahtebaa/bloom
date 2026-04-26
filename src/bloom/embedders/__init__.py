"""Pluggable embedder providers.

The default `none` embedder makes Bloom work offline with zero dependencies.
Other providers (openai, anthropic-via-voyage, local sentence-transformers)
are opt-in extras — install with e.g. `pip install bloom-mcp[openai]`.
"""

from __future__ import annotations

from bloom.config import EmbedderConfig
from bloom.embedders.base import Embedder
from bloom.embedders.none import NoneEmbedder


def load_embedder(cfg: EmbedderConfig) -> Embedder:
    """Resolve the configured embedder, importing optional deps lazily."""
    provider = (cfg.provider or "none").lower()
    if provider == "none":
        return NoneEmbedder()
    if provider == "openai":
        from bloom.embedders.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=cfg.model or "text-embedding-3-small")
    if provider == "anthropic":
        from bloom.embedders.anthropic import VoyageEmbedder

        return VoyageEmbedder(model=cfg.model or "voyage-3-lite")
    if provider == "local":
        from bloom.embedders.local import LocalEmbedder

        return LocalEmbedder(model=cfg.model or "all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder provider: {provider!r}")
