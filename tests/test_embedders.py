"""Embedder loader + no-op behavior tests.

These exercise the lazy-import + graceful-fallback contract: misconfigured
or missing embedder providers should never raise out of `load_embedder`.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from bloom.config import EmbedderConfig
from bloom.embedders import load_embedder
from bloom.embedders.none import NoneEmbedder


def test_load_embedder_none_returns_no_op() -> None:
    emb = load_embedder(EmbedderConfig(provider="none"))
    assert isinstance(emb, NoneEmbedder)
    assert emb.dim == 0
    assert emb.name == "none"


def test_load_embedder_unknown_provider_falls_back(capsys: pytest.CaptureFixture[str]) -> None:
    emb = load_embedder(EmbedderConfig(provider="invented-xyz"))
    assert isinstance(emb, NoneEmbedder)
    err = capsys.readouterr().err
    assert "invented-xyz" in err or "unknown" in err.lower()


def test_load_embedder_openai_missing_package_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the openai package isn't importable, we degrade to NoneEmbedder."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("openai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    emb = load_embedder(EmbedderConfig(provider="openai", model="text-embedding-3-small"))
    assert isinstance(emb, NoneEmbedder)
    err = capsys.readouterr().err
    assert "openai" in err.lower()


def test_load_embedder_openai_missing_api_key_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing OPENAI_API_KEY should not crash — degrade to NoneEmbedder.

    Skipped if the real `openai` package isn't installed (the import-error
    path is covered separately above).
    """
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    emb = load_embedder(EmbedderConfig(provider="openai"))
    assert isinstance(emb, NoneEmbedder)
    err = capsys.readouterr().err
    assert "openai" in err.lower()


def test_none_embedder_embed_doc_and_query_safe() -> None:
    """No-op embedder must return something safe to inspect (zero-dim array)."""
    emb = NoneEmbedder()
    d = emb.embed_doc("anything")
    q = emb.embed_query("anything")
    # Either a zero-length array or None — the contract is "callers check dim>0".
    assert getattr(d, "size", 0) == 0
    assert getattr(q, "size", 0) == 0
    assert emb.dim == 0


def test_load_embedder_voyage_alias_falls_back_without_dep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `voyage` alias should route through the anthropic adapter and
    fall back cleanly if voyageai isn't installed."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "voyageai" or name.startswith("voyageai."):
            raise ImportError("voyageai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("VOYAGE_API_KEY", "fake")

    emb = load_embedder(EmbedderConfig(provider="voyage"))
    assert isinstance(emb, NoneEmbedder)
