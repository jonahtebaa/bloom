"""MCP tool implementations.

Each function returns a JSON-serializable dict that the MCP server forwards
verbatim to the client. Keep these pure — no console I/O, no global state
besides the Database/Config passed in.
"""

from __future__ import annotations

import time
from typing import Any

from bloom.config import Config
from bloom.db import Database
from bloom.recall import ScoredTurn, recall


def _format_snippet(content: str, limit: int) -> str:
    content = content.strip()
    return content if len(content) <= limit else content[:limit] + "…"


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "tags": row["tags"] if "tags" in row.keys() else None,
        "ts": int(row["ts"] or 0),
    }


def _scored_to_dict(s: ScoredTurn, snippet_max: int) -> dict[str, Any]:
    return {
        "id": s.id,
        "session_id": s.session_id,
        "role": s.role,
        "content": _format_snippet(s.content, snippet_max),
        "tags": s.tags,
        "ts": s.ts,
        "score": round(s.score, 3),
    }


def tool_recall(
    db: Database,
    cfg: Config,
    query: str,
    k: int | None = None,
    session_filter: str | None = None,
) -> dict[str, Any]:
    """Search past turns by query — keyword + recency scored, top-k returned."""
    k = k or cfg.retrieve_top_k
    results = recall(db, query, k=k, session_id=session_filter)
    return {
        "count": len(results),
        "results": [_scored_to_dict(s, cfg.snippet_max_chars) for s in results],
    }


def tool_remember(
    db: Database,
    cfg: Config,  # noqa: ARG001
    content: str,
    session: str | None = None,
    tags: str | None = None,
    role: str | None = "note",
) -> dict[str, Any]:
    """Persist a single turn so future `recall` calls can surface it."""
    if not content or not content.strip():
        return {"ok": False, "error": "content is empty"}
    turn_id = db.insert_turn(
        content=content.strip(),
        session_id=session,
        role=role,
        tags=tags,
    )
    return {"ok": True, "id": turn_id, "ts": int(time.time())}


def tool_recent(
    db: Database,
    cfg: Config,
    session_id: str,
    n: int = 20,
) -> dict[str, Any]:
    """Return the last N turns from a specific session, in chronological order."""
    rows = db.fetch_recent(session_id, n=n)
    return {
        "session_id": session_id,
        "count": len(rows),
        "results": [
            {**_row_to_dict(r), "content": _format_snippet(r["content"] or "", cfg.snippet_max_chars)}
            for r in rows
        ],
    }


def tool_sessions(db: Database, cfg: Config, limit: int = 50) -> dict[str, Any]:  # noqa: ARG001
    rows = db.list_sessions(limit=limit)
    return {
        "count": len(rows),
        "results": [
            {
                "id": r["id"],
                "started_at": int(r["started_at"] or 0),
                "last_ts": int(r["last_ts"] or 0),
                "turn_count": int(r["turn_count"] or 0),
            }
            for r in rows
        ],
    }


def tool_forget(db: Database, cfg: Config, turn_id: int) -> dict[str, Any]:  # noqa: ARG001
    deleted = db.delete_turn(int(turn_id))
    return {"ok": deleted, "id": int(turn_id)}


def tool_stats(db: Database, cfg: Config) -> dict[str, Any]:
    s = db.stats()
    s["embedder"] = cfg.embedder.provider
    return s
