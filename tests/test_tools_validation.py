"""API-layer validation: clamping, soft-delete, unicode, size caps."""

from __future__ import annotations

from pathlib import Path

import pytest

from bloom.config import Config
from bloom.db import Database
from bloom.tools import tool_forget, tool_recall, tool_recent, tool_remember


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Database, Config]:
    cfg = Config(db_path=tmp_path / "loom.db")
    db = Database(cfg.db_path)
    return db, cfg


# ---------- recall: k clamping ---------------------------------------------


def test_recall_k_zero_clamps_to_one(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="alpha beta gamma", session="s1")
    tool_remember(db, cfg, content="alpha matters here", session="s1")
    tool_remember(db, cfg, content="alpha appears again", session="s1")
    out = tool_recall(db, cfg, query="alpha", k=0)
    assert out["ok"] is True
    # k=0 clamps to 1, so at most 1 result returned (may be 0 if no match,
    # but here we definitely have matches).
    assert out["count"] == 1


def test_recall_k_negative_clamps_to_one(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    for i in range(3):
        tool_remember(db, cfg, content=f"alpha number {i}", session="s1")
    out = tool_recall(db, cfg, query="alpha", k=-1)
    assert out["ok"] is True
    assert out["count"] == 1


def test_recall_k_huge_clamps_to_fifty(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    for i in range(80):
        tool_remember(db, cfg, content=f"alpha row {i}", session="s1")
    out = tool_recall(db, cfg, query="alpha", k=999)
    assert out["ok"] is True
    assert out["count"] <= 50


def test_recall_empty_db_returns_empty(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_recall(db, cfg, query="anything")
    assert out == {"ok": True, "count": 0, "results": []}


# ---------- recall: filter vs bias -----------------------------------------


def test_recall_filter_session_restricts_results(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    # s2 has stronger keyword density on purpose.
    tool_remember(db, cfg, content="postgres only mention", session="s1")
    for _ in range(5):
        tool_remember(db, cfg, content="postgres postgres postgres queue lock", session="s2")
    out = tool_recall(db, cfg, query="postgres queue", filter_session="s1")
    assert out["ok"] is True
    assert out["count"] >= 1
    assert all(r["session_id"] == "s1" for r in out["results"])


def test_recall_filter_session_pushes_through_limit(setup: tuple[Database, Config]) -> None:
    """Stress version of the filter_session contract.

    Before the B2 fix recall fetched FTS candidates globally then
    post-filtered AFTER the limit cap. With 250 stronger matches in s2 and
    only 1 weak match in s1, filter_session="s1" returned zero. With the
    push-down, the s1 row must still be returned.
    """
    db, cfg = setup
    # 250 strong-match rows in s2.
    for i in range(250):
        tool_remember(
            db,
            cfg,
            content=f"postgres postgres postgres queue lock row {i}",
            session="s2",
        )
    # Single weak-match row in s1.
    tool_remember(db, cfg, content="postgres mention here", session="s1")

    out = tool_recall(db, cfg, query="postgres queue", filter_session="s1", k=5)
    assert out["ok"] is True
    assert out["count"] >= 1, (
        f"filter_session push-down failed; got {out}"
    )
    assert all(r["session_id"] == "s1" for r in out["results"])


def test_recall_session_bias_boosts_but_does_not_filter(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="postgres only mention", session="s1")
    for _ in range(3):
        tool_remember(db, cfg, content="postgres postgres queue lock", session="s2")
    out = tool_recall(db, cfg, query="postgres queue", session_bias="s1", k=10)
    assert out["ok"] is True
    sessions_seen = {r["session_id"] for r in out["results"]}
    # Mixed results: both sessions present.
    assert "s1" in sessions_seen
    assert "s2" in sessions_seen
    # And s1 ranks at the top thanks to the bias.
    assert out["results"][0]["session_id"] == "s1"


# ---------- recall: unicode -------------------------------------------------


def test_recall_unicode_round_trip(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="北京 city note", session="s1")
    out = tool_recall(db, cfg, query="北京")
    assert out["ok"] is True
    assert out["count"] >= 1
    assert any("北京" in r["content"] for r in out["results"])


# ---------- remember: validation -------------------------------------------


def test_remember_empty_string_rejected(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="")
    assert out["ok"] is False
    assert "empty" in out["error"].lower()


def test_remember_whitespace_only_rejected(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="   \n\t  ")
    assert out["ok"] is False


def test_remember_oversize_rejected(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="x" * 300_000)
    assert out["ok"] is False
    assert "large" in out["error"].lower() or "262144" in out["error"]


def test_remember_role_none_defaults_to_note(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="explicit none role", session="s1", role=None)
    assert out["ok"] is True
    row = db.fetch_by_id(out["id"])
    assert row is not None
    assert row["role"] == "note"


def test_remember_returns_stored_ts(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_remember(db, cfg, content="hello", session="s1")
    assert out["ok"] is True
    row = db.fetch_by_id(out["id"])
    assert row is not None
    assert out["ts"] == int(row["ts"])


# ---------- forget: validation + soft-delete -------------------------------


def test_forget_none_returns_error_no_exception(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_forget(db, cfg, turn_id=None)
    assert out["ok"] is False
    assert "error" in out


def test_forget_non_int_returns_error(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_forget(db, cfg, turn_id="not-a-number")
    assert out["ok"] is False
    assert "error" in out


def test_forget_negative_returns_error(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_forget(db, cfg, turn_id=-3)
    assert out["ok"] is False


def test_forget_missing_id_returns_not_found(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    out = tool_forget(db, cfg, turn_id=999_999)
    assert out["ok"] is False
    assert "error" in out


def test_forget_soft_deletes_and_recall_excludes(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    written = tool_remember(db, cfg, content="forgettable postgres queue note", session="s1")
    tid = written["id"]

    # Sanity: recall finds it.
    pre = tool_recall(db, cfg, query="postgres queue")
    assert any(r["id"] == tid for r in pre["results"])

    # Soft-delete.
    forgotten = tool_forget(db, cfg, turn_id=tid)
    assert forgotten["ok"] is True

    # Recall no longer surfaces it.
    post = tool_recall(db, cfg, query="postgres queue")
    assert all(r["id"] != tid for r in post["results"])

    # Recent also excludes it.
    rec = tool_recent(db, cfg, session_id="s1")
    assert all(r["id"] != tid for r in rec["results"])

    # Row still exists physically (soft delete) but fetch_by_id filters it.
    assert db.fetch_by_id(tid) is None


# ---------- recent: n clamping ---------------------------------------------


def test_recent_n_negative_clamps_to_one(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    for i in range(5):
        tool_remember(db, cfg, content=f"row {i}", session="s1")
    out = tool_recent(db, cfg, session_id="s1", n=-5)
    assert out["ok"] is True
    assert out["count"] == 1


def test_recent_n_huge_clamps_to_two_hundred(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    for i in range(3):
        tool_remember(db, cfg, content=f"row {i}", session="s1")
    out = tool_recent(db, cfg, session_id="s1", n=99_999)
    assert out["ok"] is True
    # Only 3 actually exist, but the parameter would have been clamped to 200
    # before hitting the DB — so result is min(actual, 200).
    assert out["count"] == 3


def test_recent_n_none_uses_default(setup: tuple[Database, Config]) -> None:
    db, cfg = setup
    tool_remember(db, cfg, content="only row", session="s1")
    out = tool_recent(db, cfg, session_id="s1")
    assert out["ok"] is True
    assert out["count"] == 1
