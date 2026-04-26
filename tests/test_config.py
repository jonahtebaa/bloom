"""Config loading + write round-trip."""

from __future__ import annotations

from pathlib import Path

from bloom.config import Config, EmbedderConfig


def test_default_config_load_no_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"
    cfg = Config.load(missing)
    assert cfg.embedder.provider == "none"
    assert cfg.retrieve_top_k > 0


def test_write_then_load(tmp_path: Path) -> None:
    cfg = Config(
        db_path=tmp_path / "x.db",
        embedder=EmbedderConfig(provider="openai", model="text-embedding-3-small", api_key_env="OPENAI_API_KEY"),
        retrieve_top_k=7,
    )
    p = tmp_path / "config.toml"
    cfg.write(p)
    loaded = Config.load(p)
    assert str(loaded.db_path) == str(cfg.db_path)
    assert loaded.embedder.provider == "openai"
    assert loaded.embedder.model == "text-embedding-3-small"
    assert loaded.retrieve_top_k == 7


def test_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLOOM_DB_PATH", str(tmp_path / "from-env.db"))
    monkeypatch.setenv("BLOOM_EMBEDDER", "local")
    cfg = Config.load(tmp_path / "no-such.toml")
    assert str(cfg.db_path).endswith("from-env.db")
    assert cfg.embedder.provider == "local"
