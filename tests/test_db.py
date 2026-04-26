"""DB schema, insertion, fetching, deletion."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bloom.db import SCHEMA_VERSION, Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "loom.db")


def test_schema_initialized(db: Database) -> None:
    s = db.stats()
    assert s["schema_version"] == SCHEMA_VERSION
    assert s["turns"] == 0
    assert s["sessions"] == 0


def test_insert_and_fetch_recent(db: Database) -> None:
    sid = "test-session"
    a = db.insert_turn("first", session_id=sid, role="user", ts=1)
    b = db.insert_turn("second", session_id=sid, role="assistant", ts=2)
    rows = db.fetch_recent(sid, n=10)
    assert [r["id"] for r in rows] == [a, b]
    assert rows[0]["content"] == "first"


def test_fetch_recent_returns_newest_n_not_oldest(db: Database) -> None:
    sid = "long-session"
    for i in range(50):
        db.insert_turn(f"turn-{i}", session_id=sid, ts=1000 + i)
    rows = db.fetch_recent(sid, n=5)
    assert [r["content"] for r in rows] == [
        "turn-45",
        "turn-46",
        "turn-47",
        "turn-48",
        "turn-49",
    ]


def test_fetch_recent_caps_at_actual_count(db: Database) -> None:
    sid = "small-session"
    for i in range(3):
        db.insert_turn(f"row-{i}", session_id=sid, ts=2000 + i)
    rows = db.fetch_recent(sid, n=20)
    assert [r["content"] for r in rows] == ["row-0", "row-1", "row-2"]


def test_search_like(db: Database) -> None:
    db.insert_turn("the rain in spain")
    db.insert_turn("the lions sleep tonight")
    db.insert_turn("rain rain go away")
    rows = db.search_like(["rain"], limit=5)
    assert len(rows) == 2
    assert all("rain" in r["content"] for r in rows)


def test_delete_turn(db: Database) -> None:
    tid = db.insert_turn("ephemeral")
    assert db.delete_turn(tid) is True
    assert db.delete_turn(tid) is False


def test_list_sessions(db: Database) -> None:
    db.insert_turn("a", session_id="s1", ts=int(time.time()))
    db.insert_turn("b", session_id="s2", ts=int(time.time()))
    db.insert_turn("c", session_id="s1", ts=int(time.time()))
    sessions = db.list_sessions()
    assert {s["id"] for s in sessions} == {"s1", "s2"}
    s1 = next(s for s in sessions if s["id"] == "s1")
    assert s1["turn_count"] == 2
