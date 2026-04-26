"""Config loading + write round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from bloom.config import Config, ConfigError, EmbedderConfig, _load_env


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


def test_env_override_embedder_model_and_top_k(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLOOM_EMBEDDER", "openai")
    monkeypatch.setenv("BLOOM_EMBEDDER_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("BLOOM_RETRIEVE_TOP_K", "11")
    cfg = Config.load(tmp_path / "no-such.toml")
    assert cfg.embedder.provider == "openai"
    assert cfg.embedder.model == "text-embedding-3-large"
    assert cfg.retrieve_top_k == 11


def test_top_k_env_must_be_int(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLOOM_RETRIEVE_TOP_K", "not-a-number")
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "no-such.toml")


def test_malformed_toml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "config.toml"
    bad.write_text("[storage\nthis is = not [valid TOML")
    with pytest.raises(ConfigError) as excinfo:
        Config.load(bad)
    assert str(bad) in str(excinfo.value)


def test_load_env_sets_unset_vars(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "OPENAI_API_KEY=sk-from-env-file\n"
        "VOYAGE_API_KEY='quoted-value'\n"
        "\n"
        "BLOOM_LOG_LEVEL=DEBUG\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("BLOOM_LOG_LEVEL", raising=False)

    _load_env(env_file)

    import os

    assert os.environ.get("OPENAI_API_KEY") == "sk-from-env-file"
    assert os.environ.get("VOYAGE_API_KEY") == "quoted-value"
    assert os.environ.get("BLOOM_LOG_LEVEL") == "DEBUG"


def test_load_env_does_not_overwrite_existing(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-real-env")

    _load_env(env_file)

    import os

    assert os.environ["OPENAI_API_KEY"] == "from-real-env"


def test_config_load_pulls_env_file(monkeypatch, tmp_path: Path) -> None:
    """Config.load() should call _load_env so embedders see saved keys."""
    home = tmp_path / "bloom-home"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("VOYAGE_API_KEY=loaded-by-config\n")

    monkeypatch.setenv("BLOOM_HOME", str(home))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    Config.load(home / "no-such.toml")

    import os

    assert os.environ.get("VOYAGE_API_KEY") == "loaded-by-config"
