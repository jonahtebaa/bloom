"""Recall scoring: keyword extraction + content search + recency-weighted ranking.

The default scoring path uses zero embeddings — a deliberate choice so Bloom
works offline with no API key. If users opt into an embedder via config, the
embedder layer augments (not replaces) this scoring.
"""

from __future__ import annotations

import math
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

# Continuous runs of CJK characters become tokens of their own (CJK has no
# inter-word spaces). Covers CJK Unified Ideographs (U+4E00-U+9FFF) and
# Hangul Syllables (U+AC00-U+D7AF).
_CJK_RUN = re.compile(r"[一-鿿가-힯]+")
# Tokens matching ALL_CAPS / SNAKE_CAPS identifiers, pinned to the front of
# the keyword list because they're high-signal (code/config references).
_CAPS_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]+$")


def extract_keywords(text: str, max_k: int = 8) -> list[str]:
    """Pick the top-N salient tokens from a query string.

    Tokens are unicode word runs (`\\w+`) of length ≥2, plus continuous CJK
    runs, minus stopwords. ALL_CAPS / SNAKE_CAPS identifiers are pinned to
    the front of the result in their original order since they're high-signal.
    """
    if not text or not text.strip():
        return []
    raw_tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    # Add CJK runs as additional tokens (they often appear as single \w+ runs
    # already, but explicit handling guards against tokenizer drift).
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

    # Pin caps tokens to the front in original order, removing duplicates.
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
    candidates = db.search_content(keywords, limit=k * candidate_multiplier)
    return score_turns(candidates, keywords, session_id=session_id)[:k]
