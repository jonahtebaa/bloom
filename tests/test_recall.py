"""Recall scoring + keyword extraction."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bloom.db import Database
from bloom.recall import extract_keywords, recall


def test_extract_keywords_basic() -> None:
    kws = extract_keywords("How do I configure the OPENAI_API_KEY for embeddings?")
    assert "openai_api_key" in kws
    assert "configure" in kws
    assert "the" not in kws


def test_extract_keywords_empty() -> None:
    assert extract_keywords("") == []
    assert extract_keywords("   ") == []


def test_extract_keywords_caps_token_kept() -> None:
    kws = extract_keywords("normal words and ALL_CAPS_TOKEN here")
    assert "all_caps_token" in kws


def test_extract_keywords_caps_pinned_when_truncated() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota CAP_TOKEN"
    kws = extract_keywords(text, max_k=8)
    assert "cap_token" in kws
    assert kws[0] == "cap_token"


@pytest.fixture
def populated_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "loom.db")
    now = int(time.time())
    db.insert_turn("we decided to use postgres for the queue", session_id="s1", ts=now - 60)
    db.insert_turn("redis stays as a cache only", session_id="s1", ts=now - 120)
    db.insert_turn("apple pie recipe", session_id="s2", ts=now - 3600 * 24 * 30)
    return db


def test_recall_finds_relevant(populated_db: Database) -> None:
    results = populated_db
    out = recall(results, "what did we decide about postgres queue", k=3)
    assert len(out) >= 1
    assert any("postgres" in r.content.lower() for r in out)


def test_recall_session_bonus(populated_db: Database) -> None:
    out_default = recall(populated_db, "queue cache", k=5)
    out_biased = recall(populated_db, "queue cache", k=5, session_id="s1")
    assert all(r.session_id == "s1" for r in out_biased[:2])
    assert len(out_biased) >= len(out_default[:2])


def test_recall_empty_query(populated_db: Database) -> None:
    assert recall(populated_db, "", k=5) == []
