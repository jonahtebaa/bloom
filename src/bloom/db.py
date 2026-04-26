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
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

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
        self._con = sqlite3.connect(
            str(self.path), timeout=5.0, check_same_thread=False
        )
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield the cached connection. Kept for backwards compatibility —
        the connection lifecycle is now owned by the Database instance.
        """
        try:
            yield self._con
            self._con.commit()
        except Exception:
            self._con.rollback()
            raise

    def close(self) -> None:
        try:
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
            # Backfill any pre-existing rows.
            con.execute(
                "INSERT INTO turns_fts(rowid, content) "
                "SELECT id, content FROM turns "
                "WHERE id NOT IN (SELECT rowid FROM turns_fts)"
            )
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
                        ORDER BY ts DESC, id DESC
                        LIMIT ?
                    )
                    ORDER BY ts ASC, id ASC
                    """,
                    (session_id, n),
                ).fetchall()
            )

    def search_content(self, keywords: list[str], limit: int) -> list[sqlite3.Row]:
        """Full-text search over `turns.content` via FTS5, ordered by bm25."""
        if not keywords:
            return []
        # Build MATCH query: keywords OR'd, FTS5-special chars wrapped in quotes.
        terms = [_fts5_escape(k) for k in keywords if k]
        terms = [t for t in terms if t]
        if not terms:
            return []
        match_query = " OR ".join(terms)
        with self.connect() as con:
            try:
                return list(
                    con.execute(
                        """
                        SELECT t.id, t.session_id, t.role, t.content, t.tags, t.ts
                        FROM turns_fts f
                        JOIN turns t ON t.id = f.rowid
                        WHERE turns_fts MATCH ?
                        ORDER BY bm25(turns_fts)
                        LIMIT ?
                        """,
                        (match_query, limit),
                    ).fetchall()
                )
            except sqlite3.OperationalError:
                # Malformed query or empty FTS table — return cleanly.
                return []

    def search_like(self, keywords: list[str], limit: int) -> list[sqlite3.Row]:
        """Deprecated alias — delegates to `search_content`."""
        return self.search_content(keywords, limit)

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
                    LEFT JOIN turns t ON t.session_id = s.id
                    GROUP BY s.id
                    ORDER BY last_ts DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def delete_turn(self, turn_id: int) -> bool:
        with self.connect() as con:
            cur = con.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
            return cur.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self.connect() as con:
            turns = con.execute("SELECT COUNT(*) AS c FROM turns").fetchone()["c"]
            sessions = con.execute(
                "SELECT COUNT(DISTINCT session_id) AS c FROM turns WHERE session_id IS NOT NULL"
            ).fetchone()["c"]
            oldest_row = con.execute("SELECT MIN(ts) AS t FROM turns").fetchone()
            newest_row = con.execute("SELECT MAX(ts) AS t FROM turns").fetchone()
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
