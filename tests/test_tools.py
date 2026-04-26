"""Tool-level behavior — what the MCP server exposes to clients."""

from __future__ import annotations

from pathlib import Path

import pytest

from bloom.config import Config
from bloom.db import Database
from bloom.tools import (
    tool_forget,
    tool_recall,
    tool_recent,
    tool_remember,
    tool_sessions,
    tool_stats,
)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Database, Config]:
    cfg = Config(db_path=tmp_path / "loom.db")
    db = Database(cfg.db_path)
    return db, cfg


def test_remember_persists_turn(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="we decided X", session_id="abc", tags="decision")
    assert out["ok"] is True
    assert isinstance(out["id"], int)


def test_remember_rejects_empty(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    assert tool_remember(db, cfg, content="   ")["ok"] is False


def test_recall_round_trip(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="postgres queue with SKIP LOCKED", session_id="s1")
    tool_remember(db, cfg, content="redis stays as cache", session_id="s1")
    out = tool_recall(db, cfg, query="postgres queue")
    assert out["count"] >= 1
    assert any("postgres" in r["content"].lower() for r in out["results"])


def test_recent_chronological(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="first", session_id="s1")
    tool_remember(db, cfg, content="second", session_id="s1")
    out = tool_recent(db, cfg, session_id="s1")
    assert [r["content"] for r in out["results"]] == ["first", "second"]


def test_sessions_lists_all(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="x", session_id="s1")
    tool_remember(db, cfg, content="y", session_id="s2")
    out = tool_sessions(db, cfg)
    ids = {r["id"] for r in out["results"]}
    assert ids == {"s1", "s2"}


def test_forget_removes_turn(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    written = tool_remember(db, cfg, content="ephemeral")
    assert tool_forget(db, cfg, turn_id=written["id"])["ok"] is True
    assert tool_forget(db, cfg, turn_id=written["id"])["ok"] is False


def test_stats_shape(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="a")
    s = tool_stats(db, cfg)
    assert s["turns"] == 1
    assert s["embedder"] == "none"
    assert s["schema_version"] >= 1


def test_tool_stats_returns_all_expected_keys(setup: tuple[Database, Config]) -> None:
    """tool_stats must expose all documented keys with non-None values after
    turns have been inserted. Covers legacy aliases (`turns`, `sessions`)
    and canonical names (`total_turns`, `total_sessions`, `db_size_bytes`,
    `schema_version`, `embedder`).
    """
    db, cfg = setup
    tool_remember(db, cfg, content="a", session_id="s1")
    tool_remember(db, cfg, content="b", session_id="s1")
    tool_remember(db, cfg, content="c", session_id="s2")

    s = tool_stats(db, cfg)
    expected_keys = {
        "ok",
        "total_turns",
        "total_sessions",
        "turns",
        "sessions",
        "db_size_bytes",
        "schema_version",
        "embedder",
        "oldest_ts",
        "newest_ts",
        "db_path",
    }
    missing = expected_keys - set(s)
    assert not missing, f"tool_stats missing keys: {missing}"
    for key in (
        "ok",
        "total_turns",
        "total_sessions",
        "turns",
        "sessions",
        "db_size_bytes",
        "schema_version",
        "embedder",
    ):
        assert s[key] is not None, f"tool_stats[{key!r}] is None"

    # Legacy alias / canonical name agreement.
    assert s["total_turns"] == s["turns"] == 3
    assert s["total_sessions"] == s["sessions"] == 2
