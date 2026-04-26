"""Configuration loader for Bloom.

Resolution order (highest precedence first):
  1. Environment variables (see list below)
  2. ~/.bloom/config.toml
  3. ~/.bloom/.env (loaded into the env on Config.load() if values aren't set)
  4. Built-in defaults

Users normally run `bloom-mcp init` once to write config.toml. Env vars are
useful for CI, containers, or per-shell overrides.

Supported environment variables
--------------------------------
    BLOOM_HOME              Override ~/.bloom location
    BLOOM_DB_PATH           SQLite database path
    BLOOM_LOG_LEVEL         INFO / DEBUG / WARNING / ERROR
    BLOOM_EMBEDDER          none / openai / anthropic / local
    BLOOM_EMBEDDER_MODEL    Model name for the chosen embedder
    BLOOM_RETRIEVE_TOP_K    Default top_k for recall (int)
    OPENAI_API_KEY          Auth for the openai embedder
    VOYAGE_API_KEY          Auth for the anthropic / voyage embedder
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(Exception):
    """Raised when the user's config file is malformed or unreadable."""


def default_home() -> Path:
    """Return the Bloom home directory (~/.bloom by default)."""
    return Path(os.environ.get("BLOOM_HOME") or Path.home() / ".bloom")


def default_db_path() -> Path:
    return default_home() / "loom.db"


def default_config_path() -> Path:
    return default_home() / "config.toml"


def default_env_path() -> Path:
    return default_home() / ".env"


def _load_env(path: Path | None = None) -> None:
    """Load `~/.bloom/.env` into `os.environ` for keys that aren't already set.

    Parses simple `KEY=VALUE` lines. Comments (`#`) and blank lines are skipped.
    Values are taken verbatim — no shell escaping, no quote stripping beyond
    a single pair of surrounding double or single quotes for convenience.
    Existing env vars are NEVER overwritten — the real environment always wins.
    """
    env_path = path or default_env_path()
    if not env_path.exists():
        return
    try:
        text = env_path.read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


@dataclass
class EmbedderConfig:
    provider: str = "none"
    model: str | None = None
    api_key_env: str | None = None


@dataclass
class Config:
    db_path: Path = field(default_factory=default_db_path)
    log_level: str = "INFO"
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    retrieve_top_k: int = 5
    retrieve_max_chars: int = 4000
    snippet_max_chars: int = 600

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        # Side-effect: pull saved API keys / overrides from ~/.bloom/.env into
        # the process env so embedders that read os.environ.get(...) see them.
        _load_env()

        cfg_path = path or default_config_path()
        data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "rb") as fh:
                    data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(
                    f"could not parse {cfg_path}: {e}. "
                    "Fix the TOML syntax or delete the file and re-run "
                    "`bloom-mcp init`."
                ) from e
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        storage = data.get("storage", {})
        recall = data.get("recall", {})
        embedder = data.get("embedder", {})
        logging_cfg = data.get("logging", {})

        db_path_str = os.environ.get("BLOOM_DB_PATH") or storage.get("db_path")
        db_path = Path(db_path_str).expanduser() if db_path_str else default_db_path()

        log_level = os.environ.get("BLOOM_LOG_LEVEL") or logging_cfg.get("level", "INFO")

        provider = os.environ.get("BLOOM_EMBEDDER") or embedder.get("provider", "none")
        model = os.environ.get("BLOOM_EMBEDDER_MODEL") or embedder.get("model")
        emb = EmbedderConfig(
            provider=provider,
            model=model,
            api_key_env=embedder.get("api_key_env"),
        )

        top_k_raw = os.environ.get("BLOOM_RETRIEVE_TOP_K")
        if top_k_raw is not None:
            try:
                top_k = int(top_k_raw)
            except ValueError as e:
                raise ConfigError(
                    f"BLOOM_RETRIEVE_TOP_K must be an integer, got {top_k_raw!r}"
                ) from e
        else:
            top_k = int(recall.get("top_k", 5))

        return cls(
            db_path=db_path,
            log_level=log_level.upper(),
            embedder=emb,
            retrieve_top_k=top_k,
            retrieve_max_chars=int(recall.get("max_chars", 4000)),
            snippet_max_chars=int(recall.get("snippet_max_chars", 600)),
        )

    def write(self, path: Path | None = None) -> Path:
        cfg_path = path or default_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Bloom configuration\n"
            "# Generated by `bloom-mcp init`. Edit freely.\n"
            "\n"
            "[storage]\n"
            f'db_path = "{self.db_path}"\n'
            "\n"
            "[recall]\n"
            f"top_k = {self.retrieve_top_k}\n"
            f"max_chars = {self.retrieve_max_chars}\n"
            f"snippet_max_chars = {self.snippet_max_chars}\n"
            "\n"
            "[embedder]\n"
            f'provider = "{self.embedder.provider}"\n'
        )
        if self.embedder.model:
            body += f'model = "{self.embedder.model}"\n'
        if self.embedder.api_key_env:
            body += f'api_key_env = "{self.embedder.api_key_env}"\n'
        body += (
            "\n"
            "[logging]\n"
            f'level = "{self.log_level}"\n'
        )
        cfg_path.write_text(body)
        return cfg_path
