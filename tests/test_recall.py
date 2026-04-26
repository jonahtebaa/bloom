"""Recall scoring + keyword extraction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
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


# ---------- embedder-aware recall ------------------------------------------


class _FakeEmbedder:
    """Deterministic fake embedder for hybrid-recall tests.

    Each registered text gets a fixed unit vector. Unknown text gets a
    zero vector so cosine similarity reads 0 — which is exactly what real
    embedders shouldn't do but is fine for forcing predictable rankings.
    """

    name = "fake"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._map = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}

    def _vec(self, text: str) -> np.ndarray:
        if text in self._map:
            return self._map[text]
        return np.zeros(self.dim, dtype=np.float32)

    def embed_doc(self, content: str) -> np.ndarray:
        return self._vec(content)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


class _FailingEmbedder:
    name = "boom"
    dim = 4

    def embed_doc(self, content: str) -> np.ndarray:  # noqa: ARG002
        raise RuntimeError("doc boom")

    def embed_query(self, query: str) -> np.ndarray:  # noqa: ARG002
        raise RuntimeError("query boom")


def _store_with_embedding(
    db: Database, content: str, vec: np.ndarray, session_id: str = "s1", ts: int | None = None
) -> int:
    arr = np.asarray(vec, dtype=np.float32)
    return db.insert_turn(
        content=content,
        session_id=session_id,
        embedding=arr.tobytes(),
        ts=ts,
    )


def test_recall_with_embedder_prefers_semantic_match(tmp_path: Path) -> None:
    """B is the cosine-closest match; even though A has more keyword hits,
    the hybrid score should rank B at the top with the embedder enabled."""
    db = Database(tmp_path / "loom.db")
    now = int(time.time())

    query_text = "queue choice"
    query_vec = [1.0, 0.0, 0.0, 0.0]
    a_vec = [0.0, 1.0, 0.0, 0.0]  # orthogonal — keyword strong but cosine 0
    b_vec = [1.0, 0.0, 0.0, 0.0]  # parallel to query — cosine 1
    c_vec = [0.0, 0.0, 1.0, 0.0]  # unrelated

    a_text = "queue queue queue keyword stuffed but semantically off"
    b_text = "queue lock decision reasoning"
    c_text = "queue note unrelated"

    a_id = _store_with_embedding(db, a_text, np.asarray(a_vec), ts=now - 60)
    b_id = _store_with_embedding(db, b_text, np.asarray(b_vec), ts=now - 60)
    _store_with_embedding(db, c_text, np.asarray(c_vec), ts=now - 60)

    embedder = _FakeEmbedder({query_text: query_vec, a_text: a_vec, b_text: b_vec, c_text: c_vec})

    out = recall(db, query_text, k=3, embedder=embedder)
    ids = [r.id for r in out]
    assert ids[0] == b_id, f"expected B first (cosine 1), got {ids}"
    # A should still appear (it has keyword hits) but rank below B.
    assert a_id in ids


def test_recall_with_embedder_handles_missing_embedding(tmp_path: Path) -> None:
    """Mixed corpus: some rows have embeddings, some don't. The missing ones
    score cosine=0 but must still be returned — never crash."""
    db = Database(tmp_path / "loom.db")
    now = int(time.time())

    query_text = "queue choice"
    query_vec = [1.0, 0.0, 0.0, 0.0]

    with_emb_id = _store_with_embedding(
        db, "queue choice with embedding", np.asarray(query_vec), ts=now - 60
    )
    no_emb_id = db.insert_turn(
        content="queue choice no embedding",
        session_id="s1",
        ts=now - 30,
    )

    embedder = _FakeEmbedder(
        {query_text: query_vec, "queue choice with embedding": query_vec}
    )

    out = recall(db, query_text, k=5, embedder=embedder)
    ids = [r.id for r in out]
    assert with_emb_id in ids
    assert no_emb_id in ids
    # The one with the matching embedding should rank above the one without.
    assert ids.index(with_emb_id) < ids.index(no_emb_id)


def test_recall_with_failing_embedder_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If embed_query raises, recall must degrade to keyword-only ranking."""
    db = Database(tmp_path / "loom.db")
    now = int(time.time())
    db.insert_turn("postgres queue lock decision", session_id="s1", ts=now - 60)
    db.insert_turn("redis cache only", session_id="s1", ts=now - 120)

    out = recall(db, "postgres queue", k=3, embedder=_FailingEmbedder())
    assert any("postgres" in r.content for r in out)

    # And the failure was logged to stderr (not raised).
    err = capsys.readouterr().err
    assert "embed_query failed" in err
