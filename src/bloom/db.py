"""SQLite-backed storage for Bloom turns and sessions.

Schema is intentionally minimal — turns are the only first-class entity.
Sessions are derived from `session_id` on turns; we keep a sessions table for
metadata (started_at, ended_at, label) but it's optional.

Migrations are handled via a single `schema_version` row in `bloom_meta`.
Bumping `SCHEMA_VERSION` and adding a new branch in `_migrate` is the way
forward — never drop columns in place.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class Database:
    """Thin wrapper around sqlite3 with Bloom-specific helpers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=5.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

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
        con.execute(
            "INSERT OR REPLACE INTO bloom_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
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
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT id, session_id, role, content, tags, ts
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY ts ASC
                    LIMIT ?
                    """,
                    (session_id, n),
                ).fetchall()
            )

    def search_like(self, keywords: list[str], limit: int) -> list[sqlite3.Row]:
        if not keywords:
            return []
        clause = " OR ".join(["content LIKE ?"] * len(keywords))
        params: list[Any] = [f"%{k}%" for k in keywords]
        params.append(limit)
        with self.connect() as con:
            return list(
                con.execute(
                    f"""
                    SELECT id, session_id, role, content, tags, ts
                    FROM turns
                    WHERE {clause}
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            )

    def list_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT
                      session_id AS id,
                      MIN(ts)    AS started_at,
                      MAX(ts)    AS last_ts,
                      COUNT(*)   AS turn_count
                    FROM turns
                    WHERE session_id IS NOT NULL
                    GROUP BY session_id
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
