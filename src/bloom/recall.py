"""Recall scoring: keyword extraction + content search + recency-weighted ranking.

The default scoring path uses zero embeddings — a deliberate choice so Bloom
works offline with no API key. When an embedder is configured, recall
re-ranks the FTS5 candidate pool using cosine similarity:

    final = 0.4 * bm25_norm + 0.5 * cosine_sim + 0.1 * recency_norm

The cosine weight slightly outranks bm25 so semantically-strong matches
beat raw keyword density on non-exact queries, while exact-keyword queries
still surface (FTS5 already filtered out non-matches before re-rank).

Embedding the query is a single network/CPU call per recall (not per row);
candidate document embeddings are fetched from the DB where they were
written at `remember`-time. Failures in the embedding call (timeout, API
error, missing dep) fall back cleanly to keyword-only ranking.
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bloom.db import Database

if TYPE_CHECKING:
    import numpy as np

    from bloom.embedders.base import Embedder

_STOPWORDS = {
    "the", "and", "for", "but", "with", "you", "your", "are", "was", "were",
    "this", "that", "these", "those", "have", "has", "had", "they", "them",
    "their", "from", "into", "about", "what", "when", "where", "why", "how",
    "can", "will", "would", "could", "should", "did", "does", "doing", "been",
    "any", "all", "some", "more", "most", "other", "such", "than", "then",
    "out", "off", "not", "now", "one", "two", "way", "use", "get", "got",
}

# Continuous runs of CJK characters become tokens of their own (CJK has no
# inter-word spaces). Covers CJK Unified Ideographs (U+4E00-U+9FFF) and
# Hangul Syllables (U+AC00-U+D7AF).
_CJK_RUN = re.compile(r"[一-鿿가-힯]+")
# Tokens matching ALL_CAPS / SNAKE_CAPS identifiers, pinned to the front of
# the keyword list because they're high-signal (code/config references).
_CAPS_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]+$")

# Embedder candidate pool — wider than the default keyword multiplier so the
# cosine re-rank has enough variety to actually move things around.
_EMBEDDER_CANDIDATE_LIMIT = 100


def extract_keywords(text: str, max_k: int = 8) -> list[str]:
    """Pick the top-N salient tokens from a query string.

    Tokens are unicode word runs (`\\w+`) of length ≥2, plus continuous CJK
    runs, minus stopwords. ALL_CAPS / SNAKE_CAPS identifiers are pinned to
    the front of the result in their original order since they're high-signal.
    """
    if not text or not text.strip():
        return []
    raw_tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    raw_tokens.extend(_CJK_RUN.findall(text))

    seen: list[str] = []
    seen_lc: set[str] = set()
    caps_in_order: list[str] = []
    caps_seen: set[str] = set()

    for w in raw_tokens:
        if len(w) < 2:
            continue
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        if _CAPS_TOKEN.match(w) and lw not in caps_seen:
            caps_in_order.append(lw)
            caps_seen.add(lw)
        if lw in seen_lc:
            continue
        seen.append(lw)
        seen_lc.add(lw)

    for c in caps_in_order:
        if c in seen:
            seen.remove(c)
    final = caps_in_order + seen
    return final[:max_k]


@dataclass
class ScoredTurn:
    id: int
    session_id: str | None
    role: str | None
    content: str
    ts: int
    score: float
    tags: str | None = None


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity of two 1D float vectors. Returns 0.0 if either is empty
    or if either has zero norm."""
    import numpy as np

    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _safe_query_embed(embedder: "Embedder", query: str) -> "np.ndarray | None":
    """Call embedder.embed_query; on any exception, log and return None."""
    try:
        return embedder.embed_query(query)
    except Exception as e:  # noqa: BLE001 — never let embedding break recall.
        print(f"[bloom] embed_query failed: {e}", file=sys.stderr)
        return None


def score_turns(
    rows: list[sqlite3.Row],
    keywords: list[str],
    session_id: str | None = None,
    now: int | None = None,
) -> list[ScoredTurn]:
    """Score candidate rows by keyword hits + same-session bonus + recency.

    This is the keyword-only path used when no embedder is configured, or
    when query embedding fails. Embedder-aware scoring lives in `recall()`.
    """
    if not rows or not keywords:
        return []
    now = now or int(time.time())
    scored: list[ScoredTurn] = []
    for r in rows:
        content_lc = (r["content"] or "").lower()
        hits = sum(1 for k in keywords if k in content_lc)
        if hits == 0:
            continue
        score = float(hits)
        if session_id and r["session_id"] == session_id:
            score += 2.0
        age_days = max(0.0, (now - int(r["ts"] or now)) / 86400.0)
        # Exponential decay (half-life ~9.7d) so older still scores lower
        # but never collapses to zero — yesterday and last-year shouldn't tie.
        score += 2.0 * math.exp(-age_days / 14.0)
        scored.append(
            ScoredTurn(
                id=int(r["id"]),
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"] or "",
                ts=int(r["ts"] or 0),
                score=score,
                tags=r["tags"] if "tags" in r.keys() else None,  # noqa: SIM118
            )
        )
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def _score_with_embedder(
    rows: list[sqlite3.Row],
    keywords: list[str],
    query_vec: "np.ndarray",
    id_to_blob: dict[int, bytes],
    embedder_dim: int,
    session_id: str | None,
    now: int,
) -> list[ScoredTurn]:
    """Hybrid score: 0.4*bm25_norm + 0.5*cosine + 0.1*recency_norm.

    bm25 isn't directly exposed by sqlite3's FTS5 binding through our query,
    but the FTS rowset arrives in bm25-rank order. We approximate the
    normalized bm25 contribution by using the row's *position* in the
    candidate list (best=1.0, worst=0.0). Same for recency.
    """
    import numpy as np

    if not rows:
        return []
    n = len(rows)
    # Position-based bm25 proxy: top of candidate list = 1.0, bottom = 0.0.
    # FTS5 already sorted by bm25 ascending (smaller is better in bm25), so
    # the first row is the strongest match.
    bm25_norm = [1.0 - (i / max(1, n - 1)) for i in range(n)] if n > 1 else [1.0]

    # Recency normalization across this candidate set only.
    ages = [max(0.0, (now - int(r["ts"] or now)) / 86400.0) for r in rows]
    rec_raw = [math.exp(-a / 14.0) for a in ages]
    max_rec = max(rec_raw) if rec_raw else 1.0
    rec_norm = [r / max_rec for r in rec_raw] if max_rec > 0 else [0.0] * n

    scored: list[ScoredTurn] = []
    for i, r in enumerate(rows):
        rid = int(r["id"])
        blob = id_to_blob.get(rid)
        cosine = 0.0
        if blob is not None:
            try:
                doc_vec = np.frombuffer(blob, dtype=np.float32)
                if doc_vec.size == embedder_dim:
                    cosine = _cosine(query_vec, doc_vec)
            except Exception:  # noqa: BLE001 — bad blob, skip cosine.
                cosine = 0.0

        # Same-session soft bias: small additive nudge so it doesn't dominate
        # the normalized hybrid score.
        bias = 0.05 if (session_id and r["session_id"] == session_id) else 0.0

        # Hybrid: cosine slightly dominates so semantically-strong matches
        # beat raw keyword density; bm25 still anchors exact-match queries.
        score = 0.4 * bm25_norm[i] + 0.5 * cosine + 0.1 * rec_norm[i] + bias
        scored.append(
            ScoredTurn(
                id=rid,
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"] or "",
                ts=int(r["ts"] or 0),
                score=score,
                tags=r["tags"] if "tags" in r.keys() else None,  # noqa: SIM118
            )
        )

    # Filter to rows that have at least *some* keyword overlap — prevents
    # cosine-only matches from polluting results when the user typed a
    # specific keyword query. (Mirrors `score_turns`'s hits>0 gate.)
    kw_filtered: list[ScoredTurn] = []
    for s in scored:
        cl = (s.content or "").lower()
        if not keywords or any(k in cl for k in keywords):
            kw_filtered.append(s)
    kw_filtered.sort(key=lambda x: x.score, reverse=True)
    return kw_filtered


def recall(
    db: Database,
    query: str,
    k: int = 5,
    session_id: str | None = None,
    candidate_multiplier: int = 3,
    filter_session: str | None = None,
    embedder: Any | None = None,
) -> list[ScoredTurn]:
    """End-to-end recall: extract keywords, fetch candidates, score, return top-k.

    `session_id` boosts results from that session (soft preference). `filter_session`
    hard-filters candidates so only that session's turns are considered.
    `embedder` (optional): if provided and `embedder.dim > 0`, augments scoring
    with cosine similarity over stored document embeddings.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return []

    use_embedder = embedder is not None and getattr(embedder, "dim", 0) > 0

    fetch_limit = k * candidate_multiplier
    if filter_session:
        fetch_limit = max(fetch_limit, 200)
    if use_embedder:
        # Widen the pool so the cosine re-rank has enough material.
        fetch_limit = max(fetch_limit, _EMBEDDER_CANDIDATE_LIMIT)

    candidates = db.search_content(keywords, limit=fetch_limit)
    if filter_session:
        candidates = [r for r in candidates if r["session_id"] == filter_session]

    if not use_embedder:
        return score_turns(candidates, keywords, session_id=session_id)[:k]

    query_vec = _safe_query_embed(embedder, query)
    if query_vec is None or query_vec.size == 0:
        # Embedding failed — degrade gracefully to keyword-only scoring.
        return score_turns(candidates, keywords, session_id=session_id)[:k]

    id_to_blob = db.fetch_embeddings([int(r["id"]) for r in candidates])
    now = int(time.time())
    scored = _score_with_embedder(
        candidates,
        keywords,
        query_vec,
        id_to_blob,
        embedder_dim=int(getattr(embedder, "dim", 0)),
        session_id=session_id,
        now=now,
    )
    return scored[:k]
