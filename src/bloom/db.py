"""SQLite-backed storage for Bloom turns and sessions.

Schema is intentionally minimal — turns are the only first-class entity.
Sessions are tracked via a `sessions` table populated by `insert_turn`; we
read from it (joined against `turns` for accurate counts) in `list_sessions`
to scale better than a per-call `GROUP BY` on large `turns` tables.

Migrations are handled via a single `schema_version` row in `bloom_meta`.
Bumping `SCHEMA_VERSION` and adding a new branch in `_migrate` is the way
forward — never drop columns in place.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

# FTS5 special characters that require quoting when used in MATCH queries.
_FTS5_SPECIAL = re.compile(r"[^A-Za-z0-9_]")


def _fts5_escape(token: str) -> str:
    """Wrap a token in double quotes (escaping internal quotes) if it contains
    FTS5-special characters; otherwise return as-is.
    """
    if not token:
        return ""
    if _FTS5_SPECIAL.search(token):
        return '"' + token.replace('"', '""') + '"'
    return token


class Database:
    """Thin wrapper around sqlite3 with Bloom-specific helpers.

    Holds a single long-lived connection on the instance so PRAGMAs apply
    once and multiple calls don't pay reconnection cost.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3.Connection is NOT thread-safe even for concurrent reads when
        # shared across threads with check_same_thread=False — every method
        # below serializes connection use with self._lock so threaded callers
        # can't interleave executes and corrupt the cursor state.
        self._lock = threading.RLock()
        self._con = sqlite3.connect(
            str(self.path), timeout=5.0, check_same_thread=False
        )
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield the cached connection under the instance lock.

        Acquires `self._lock` for the duration of the block so concurrent
        callers from multiple threads never share a cursor mid-execute.
        Re-entrant via RLock so methods that nest `with self.connect()`
        (e.g. soft_delete_turn) don't deadlock.
        """
        with self._lock:
            try:
                yield self._con
                self._con.commit()
            except Exception:
                self._con.rollback()
                raise

    def close(self) -> None:
        try:
            with self._lock:
                self._con.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS bloom_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            current = self._get_version(con)
            if current is None:
                self._migrate(con, from_version=0)
            elif current < SCHEMA_VERSION:
                self._migrate(con, from_version=current)

    @staticmethod
    def _get_version(con: sqlite3.Connection) -> int | None:
        row = con.execute(
            "SELECT value FROM bloom_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _migrate(self, con: sqlite3.Connection, from_version: int) -> None:
        if from_version < 1:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT NOT NULL,
                    tags TEXT,
                    embedding BLOB,
                    ts INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);
                CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    started_at INTEGER NOT NULL,
                    ended_at INTEGER
                );
                """
            )
        if from_version < 2:
            # FTS5 virtual table mirroring `turns.content` + triggers + backfill.
            con.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    content,
                    content='turns',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.id, old.content);
                    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
                END;
                """
            )
            # Backfill any pre-existing rows. We use FTS5's built-in 'rebuild'
            # command rather than a per-row INSERT...WHERE NOT IN guard — for
            # external-content FTS5 tables, that NOT IN against `turns_fts`
            # does not behave as you'd hope (rowids without indexed terms still
            # appear as NOT IN matches), which silently leaves pre-existing
            # v1 rows unsearchable after migration. 'rebuild' reads from the
            # external content table (turns) and reconstructs the index.
            con.execute("INSERT INTO turns_fts(turns_fts) VALUES('rebuild')")
        if from_version < 3:
            # Soft-delete column. Existing rows default to NULL (live).
            cols = {
                row["name"]
                for row in con.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "deleted_at" not in cols:
                con.execute("ALTER TABLE turns ADD COLUMN deleted_at INTEGER")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_deleted_at ON turns(deleted_at)"
            )

        # Belt-and-braces: if turns_fts exists but is empty while turns has
        # rows (DB built by an older Bloom that hit the broken NOT IN
        # backfill, or hand-rolled), rebuild the FTS index from the
        # external content table so recall doesn't silently return zero.
        try:
            fts_count = con.execute(
                "SELECT count(*) AS c FROM turns_fts"
            ).fetchone()["c"]
            turns_count = con.execute(
                "SELECT count(*) AS c FROM turns"
            ).fetchone()["c"]
            if turns_count > 0 and fts_count == 0:
                con.execute(
                    "INSERT INTO turns_fts(turns_fts) VALUES('rebuild')"
                )
        except sqlite3.OperationalError:
            # turns_fts didn't exist (shouldn't happen — migration above
            # creates it), or rebuild failed. Skip rather than crash startup.
            pass

        con.execute(
            "INSERT OR REPLACE INTO bloom_meta (key, value) VALUES ('schema_version', ?)",
            (int(SCHEMA_VERSION),),
        )

    def insert_turn(
        self,
        content: str,
        session_id: str | None = None,
        role: str | None = None,
        tags: str | None = None,
        embedding: bytes | None = None,
        ts: int | None = None,
    ) -> int:
        ts = ts or int(time.time())
        with self.connect() as con:
            cur = con.execute(
                """
                INSERT INTO turns (session_id, role, content, tags, embedding, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tags, embedding, ts),
            )
            if session_id:
                con.execute(
                    """
                    INSERT INTO sessions (id, started_at)
                    VALUES (?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (session_id, ts),
                )
            return int(cur.lastrowid)

    def update_embedding(self, turn_id: int, blob: bytes | None) -> bool:
        """Backfill or replace the embedding bytes for a turn. Returns True on hit."""
        with self.connect() as con:
            cur = con.execute(
                "UPDATE turns SET embedding = ? WHERE id = ?",
                (blob, int(turn_id)),
            )
            return cur.rowcount > 0

    def fetch_embeddings(self, turn_ids: list[int]) -> dict[int, bytes]:
        """Return `{id: embedding_bytes}` for the given ids, skipping nulls.

        Pure data-shovel: does NOT deserialize or score. Callers (recall) own
        the numpy/cosine math so this module stays numpy-free.
        """
        if not turn_ids:
            return {}
        # Chunk to stay under SQLite's variable limit on huge candidate pools.
        out: dict[int, bytes] = {}
        with self.connect() as con:
            for i in range(0, len(turn_ids), 500):
                chunk = turn_ids[i : i + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = con.execute(
                    f"SELECT id, embedding FROM turns "
                    f"WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                    [int(t) for t in chunk],
                ).fetchall()
                for r in rows:
                    blob = r["embedding"]
                    if blob is not None:
                        out[int(r["id"])] = bytes(blob)
        return out

    def iter_missing_embeddings(self, batch_size: int = 100) -> Iterator[list[sqlite3.Row]]:
        """Yield batches of live rows whose `embedding` is NULL.

        Used by the `backfill-embeddings` CLI command. Yields `(id, content)`
        rows in id order; caller passes them through an embedder and writes
        results back via `update_embedding`.
        """
        last_id = 0
        with self.connect() as con:
            while True:
                rows = list(
                    con.execute(
                        """
                        SELECT id, content
                        FROM turns
                        WHERE embedding IS NULL
                          AND deleted_at IS NULL
                          AND id > ?
                        ORDER BY id ASC
                        LIMIT ?
                        """,
                        (last_id, int(batch_size)),
                    ).fetchall()
                )
                if not rows:
                    return
                yield rows
                last_id = int(rows[-1]["id"])

    def count_missing_embeddings(self) -> int:
        """Total live turns with a NULL embedding — for backfill progress display."""
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS c FROM turns "
                "WHERE embedding IS NULL AND deleted_at IS NULL"
            ).fetchone()
            return int(row["c"])

    def fetch_recent(self, session_id: str, n: int = 20) -> list[sqlite3.Row]:
        """Return the most recent N turns for a session, sorted chronologically."""
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT * FROM (
                        SELECT id, session_id, role, content, tags, ts
                        FROM turns
                        WHERE session_id = ?
                          AND deleted_at IS NULL
                        ORDER BY ts DESC, id DESC
                        LIMIT ?
                    )
                    ORDER BY ts ASC, id ASC
                    """,
                    (session_id, n),
                ).fetchall()
            )

    def fetch_by_id(self, turn_id: int) -> sqlite3.Row | None:
        """Return a single live turn row by id, or None if missing/soft-deleted."""
        with self.connect() as con:
            return con.execute(
                """
                SELECT id, session_id, role, content, tags, ts
                FROM turns
                WHERE id = ? AND deleted_at IS NULL
                """,
                (int(turn_id),),
            ).fetchone()

    def search_content(
        self,
        keywords: list[str],
        limit: int,
        session_filter: str | None = None,
    ) -> list[sqlite3.Row]:
        """Full-text search over `turns.content` via FTS5, ordered by bm25.

        When `session_filter` is set, the WHERE clause is pushed into the
        SQL so the LIMIT applies AFTER the session filter — preventing the
        bug where N strong matches in another session displace the only
        matching row in the requested session.
        """
        if not keywords:
            return []
        # Build MATCH query: keywords OR'd, FTS5-special chars wrapped in quotes.
        terms = [_fts5_escape(k) for k in keywords if k]
        terms = [t for t in terms if t]
        if not terms:
            return []
        match_query = " OR ".join(terms)
        sql = (
            "SELECT t.id, t.session_id, t.role, t.content, t.tags, t.ts "
            "FROM turns_fts f "
            "JOIN turns t ON t.id = f.rowid "
            "WHERE turns_fts MATCH ? "
            "  AND t.deleted_at IS NULL"
        )
        params: list[Any] = [match_query]
        if session_filter is not None:
            sql += " AND t.session_id = ?"
            params.append(session_filter)
        sql += " ORDER BY bm25(turns_fts) LIMIT ?"
        params.append(limit)
        with self.connect() as con:
            try:
                return list(con.execute(sql, params).fetchall())
            except sqlite3.OperationalError:
                # Malformed query or empty FTS table — return cleanly.
                return []

    def search_like(
        self,
        keywords: list[str],
        limit: int,
        session_filter: str | None = None,
    ) -> list[sqlite3.Row]:
        """Deprecated alias — delegates to `search_content`."""
        return self.search_content(keywords, limit, session_filter=session_filter)

    def fetch_recent_with_embeddings(
        self,
        limit: int = 200,
        exclude_session: str | None = None,
        only_session: str | None = None,
    ) -> list[sqlite3.Row]:
        """Return the most recent N live rows that have an embedding stored.

        Used by recall's semantic candidate path so cosine similarity can
        surface keyword-miss matches (e.g. "queue choice" finds "we picked
        postgres for the lock primitive"). Ordered by `ts DESC, id DESC`.

        `only_session` (optional): hard-filter to a single session_id.
        `exclude_session` (optional): drop rows from a particular session.
        """
        sql = (
            "SELECT id, session_id, role, content, tags, ts, embedding "
            "FROM turns "
            "WHERE deleted_at IS NULL AND embedding IS NOT NULL"
        )
        params: list[Any] = []
        if only_session is not None:
            sql += " AND session_id = ?"
            params.append(only_session)
        if exclude_session is not None:
            sql += " AND (session_id IS NULL OR session_id != ?)"
            params.append(exclude_session)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self.connect() as con:
            return list(con.execute(sql, params).fetchall())

    def list_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        # Read from `sessions` table joined with COUNT(*) over turns for accurate counts.
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT
                      s.id        AS id,
                      s.started_at AS started_at,
                      COALESCE(MAX(t.ts), s.started_at) AS last_ts,
                      COUNT(t.id) AS turn_count
                    FROM sessions s
                    LEFT JOIN turns t
                      ON t.session_id = s.id AND t.deleted_at IS NULL
                    GROUP BY s.id
                    ORDER BY last_ts DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def delete_turn(self, turn_id: int) -> bool:
        """Hard delete — kept for callers that want it. Tools use soft_delete_turn."""
        with self.connect() as con:
            cur = con.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
            return cur.rowcount > 0

    def soft_delete_turn(self, turn_id: int, ts: int | None = None) -> bool:
        """Mark a turn as deleted without removing the row. Also evicts from FTS
        so soft-deleted turns are invisible to keyword search.
        """
        ts = ts or int(time.time())
        with self.connect() as con:
            cur = con.execute(
                """
                UPDATE turns SET deleted_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (ts, int(turn_id)),
            )
            if cur.rowcount > 0:
                # The UPDATE trigger re-inserts into turns_fts, so we explicitly
                # delete the FTS row to keep soft-deleted content out of search.
                con.execute(
                    "INSERT INTO turns_fts(turns_fts, rowid, content) "
                    "SELECT 'delete', id, content FROM turns WHERE id = ?",
                    (int(turn_id),),
                )
                return True
            return False

    def stats(self) -> dict[str, Any]:
        with self.connect() as con:
            turns = con.execute(
                "SELECT COUNT(*) AS c FROM turns WHERE deleted_at IS NULL"
            ).fetchone()["c"]
            sessions = con.execute(
                "SELECT COUNT(DISTINCT session_id) AS c FROM turns "
                "WHERE session_id IS NOT NULL AND deleted_at IS NULL"
            ).fetchone()["c"]
            oldest_row = con.execute(
                "SELECT MIN(ts) AS t FROM turns WHERE deleted_at IS NULL"
            ).fetchone()
            newest_row = con.execute(
                "SELECT MAX(ts) AS t FROM turns WHERE deleted_at IS NULL"
            ).fetchone()
            db_size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "turns": int(turns),
            "sessions": int(sessions),
            "oldest_ts": oldest_row["t"],
            "newest_ts": newest_row["t"],
            "db_size_bytes": db_size,
            "db_path": str(self.path),
            "schema_version": SCHEMA_VERSION,
        }
