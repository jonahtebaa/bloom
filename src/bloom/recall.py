"""Recall scoring: keyword extraction + LIKE search + recency-weighted ranking.

The default scoring path uses zero embeddings — a deliberate choice so Bloom
works offline with no API key. If users opt into an embedder via config, the
embedder layer augments (not replaces) this scoring.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

from bloom.db import Database

_STOPWORDS = {
    "the", "and", "for", "but", "with", "you", "your", "are", "was", "were",
    "this", "that", "these", "those", "have", "has", "had", "they", "them",
    "their", "from", "into", "about", "what", "when", "where", "why", "how",
    "can", "will", "would", "could", "should", "did", "does", "doing", "been",
    "any", "all", "some", "more", "most", "other", "such", "than", "then",
    "out", "off", "not", "now", "one", "two", "way", "use", "get", "got",
}


def extract_keywords(text: str, max_k: int = 8) -> list[str]:
    """Pick the top-N salient tokens from a query string.

    Words must be ≥3 characters, alphanumeric (with underscore), and not in
    the stopword list. ALL_CAPS or snake_case identifiers (often code/secret
    references) are pinned to the front since they're high-signal.
    """
    if not text:
        return []
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    seen: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or lw in seen:
            continue
        seen.append(lw)
        if len(seen) >= max_k:
            break
    caps = [w for w in words if (w.isupper() and len(w) >= 4) or "_" in w]
    for c in caps:
        if c.lower() not in seen:
            seen.insert(0, c.lower())
    return seen[:max_k]


@dataclass
class ScoredTurn:
    id: int
    session_id: str | None
    role: str | None
    content: str
    ts: int
    score: float
    tags: str | None = None


def score_turns(
    rows: list[sqlite3.Row],
    keywords: list[str],
    session_id: str | None = None,
    now: int | None = None,
) -> list[ScoredTurn]:
    """Score candidate rows by keyword hits + same-session bonus + recency."""
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
        age_hr = max(0.0, (now - int(r["ts"] or now)) / 3600.0)
        score += max(0.0, 2.0 - age_hr * 0.1)
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


def recall(
    db: Database,
    query: str,
    k: int = 5,
    session_id: str | None = None,
    candidate_multiplier: int = 3,
) -> list[ScoredTurn]:
    """End-to-end recall: extract keywords, fetch candidates, score, return top-k."""
    keywords = extract_keywords(query)
    if not keywords:
        return []
    candidates = db.search_like(keywords, limit=k * candidate_multiplier)
    return score_turns(candidates, keywords, session_id=session_id)[:k]
