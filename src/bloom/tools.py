"""MCP tool implementations.

Each function returns a JSON-serializable dict that the MCP server forwards
verbatim to the client. Keep these pure — no console I/O, no global state
besides the Database/Config/embedder passed in.

The optional `embedder` argument is dependency-injected by the server. When
absent, the data path matches the no-op embedder: keyword recall, no
embedding columns written.
"""

from __future__ import annotations

import sys
from typing import Any

from bloom.config import Config
from bloom.db import Database
from bloom.recall import ScoredTurn, recall

# Hard limits — enforced at the API layer so a misconfigured client can't
# blow up the DB or stuff arbitrary blobs into memory.
_MAX_CONTENT_BYTES = 256 * 1024  # 256 KB
_MAX_K = 50
_MAX_RECENT = 200
_MAX_SESSIONS = 100
_DEFAULT_ROLE = "note"


def _format_snippet(content: str, limit: int) -> str:
    content = content.strip()
    return content if len(content) <= limit else content[:limit] + "…"


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "tags": row["tags"] if "tags" in row.keys() else None,  # noqa: SIM118
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


def _embed_doc_safely(embedder: Any | None, content: str) -> bytes | None:
    """Compute a document embedding, swallow failures.

    NEVER raises — embedder issues must not fail a `remember` call. Returns
    raw float32 bytes ready to drop into the SQLite BLOB column, or None if
    no embedder is configured / dim is 0 / the call failed.
    """
    if embedder is None or getattr(embedder, "dim", 0) <= 0:
        return None
    try:
        vec = embedder.embed_doc(content)
    except Exception as e:  # noqa: BLE001
        print(f"[bloom] embed_doc failed: {e}", file=sys.stderr)
        return None
    if vec is None:
        return None
    try:
        # Lazy numpy: we only get here when an embedder is wired in, which
        # implies numpy is already present (the embedder uses it).
        import numpy as np

        arr = np.asarray(vec, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr.tobytes()
    except Exception as e:  # noqa: BLE001
        print(f"[bloom] embed_doc serialization failed: {e}", file=sys.stderr)
        return None


def tool_recall(
    db: Database,
    cfg: Config,
    query: str,
    k: int | None = None,
    session_bias: str | None = None,
    filter_session: str | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Search past turns by query — keyword (+ optional cosine) + recency, top-k.

    `session_bias` (optional): boost results from that session.
    `filter_session` (optional): restrict results to that session only.
    `embedder` (optional): when configured (dim > 0), re-rank candidates
    with cosine similarity against stored document embeddings.
    """
    if k is None:
        k = cfg.retrieve_top_k
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = cfg.retrieve_top_k
    k = max(1, min(k, _MAX_K))

    bias = session_bias if session_bias else None
    flt = filter_session if filter_session else None

    results = recall(
        db,
        query,
        k=k,
        session_id=bias,
        filter_session=flt,
        embedder=embedder,
    )
    return {
        "ok": True,
        "count": len(results),
        "results": [_scored_to_dict(s, cfg.snippet_max_chars) for s in results],
    }


def tool_remember(
    db: Database,
    cfg: Config,  # noqa: ARG001
    content: str,
    session: str | None = None,
    tags: str | None = None,
    role: str | None = _DEFAULT_ROLE,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Persist a single turn so future `recall` calls can surface it.

    If an embedder is configured (`embedder.dim > 0`), the document vector
    is computed at write-time and stored in the row's `embedding` BLOB.
    Embedder failures NEVER fail the remember call — we log to stderr and
    insert without an embedding. The row can be backfilled later with
    `bloom-mcp backfill-embeddings`.
    """
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "error": "content is empty"}
    encoded_len = len(content.encode("utf-8"))
    if encoded_len > _MAX_CONTENT_BYTES:
        return {
            "ok": False,
            "error": f"content too large (max {_MAX_CONTENT_BYTES} bytes)",
        }
    # Honor explicit role=None by falling back to default.
    effective_role = role if role else _DEFAULT_ROLE
    stripped = content.strip()
    embedding_bytes = _embed_doc_safely(embedder, stripped)
    turn_id = db.insert_turn(
        content=stripped,
        session_id=session,
        role=effective_role,
        tags=tags,
        embedding=embedding_bytes,
    )
    row = db.fetch_by_id(turn_id)
    ts = int(row["ts"]) if row else 0
    return {"ok": True, "id": turn_id, "ts": ts}


def tool_recent(
    db: Database,
    cfg: Config,
    session_id: str,
    n: int | None = None,
) -> dict[str, Any]:
    """Return the last N turns from a specific session, in chronological order."""
    if n is None:
        n = cfg.retrieve_top_k
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = cfg.retrieve_top_k
    n = max(1, min(n, _MAX_RECENT))
    rows = db.fetch_recent(session_id, n=n)
    return {
        "ok": True,
        "session_id": session_id,
        "count": len(rows),
        "results": [
            {**_row_to_dict(r), "content": _format_snippet(r["content"] or "", cfg.snippet_max_chars)}
            for r in rows
        ],
    }


def tool_sessions(db: Database, cfg: Config, limit: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    if limit is None:
        limit = 50
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, _MAX_SESSIONS))
    rows = db.list_sessions(limit=limit)
    return {
        "ok": True,
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


def tool_forget(db: Database, cfg: Config, turn_id: Any = None) -> dict[str, Any]:  # noqa: ARG001
    """Soft-delete a turn by id. Idempotent: deleting an already-deleted or
    nonexistent turn returns ok:False with a clear error.
    """
    if turn_id is None:
        return {"ok": False, "error": "turn_id is required"}
    try:
        tid = int(turn_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "turn_id must be an integer"}
    if tid < 1:
        return {"ok": False, "error": "turn_id must be a positive integer"}
    deleted = db.soft_delete_turn(tid)
    if not deleted:
        return {"ok": False, "id": tid, "error": "turn not found"}
    return {"ok": True, "id": tid}


def tool_stats(db: Database, cfg: Config) -> dict[str, Any]:
    s = db.stats()
    s["embedder"] = cfg.embedder.provider
    s["ok"] = True
    return s
