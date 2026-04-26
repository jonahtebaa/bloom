# Architecture

Bloom is intentionally small. The whole point is that you can read every line in
an evening and trust nothing weird is happening with your data.

## Components

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Code (or any MCP client)                             │
└─────────────────────────┬────────────────────────────────────┘
                          │  stdio (JSON-RPC, MCP protocol)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  bloom-mcp serve                                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ server.py     — MCP server, handles tool calls         │ │
│   │ tools.py      — recall / remember / recent / sessions /│ │
│   │                 forget / stats                         │ │
│   │ recall.py     — keyword extraction + scoring           │ │
│   │ db.py         — SQLite wrapper, schema migrations      │ │
│   │ config.py     — TOML + env vars                        │ │
│   │ embedders/    — none | openai | voyage | local         │ │
│   └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
                 ~/.bloom/loom.db (SQLite)
```

## Data model

Three tables, one of which is metadata-only.

### `turns`
Every memory is a turn. The shape:

| Column      | Type    | Notes                                              |
|-------------|---------|----------------------------------------------------|
| `id`        | INTEGER | Auto-increment primary key                         |
| `session_id`| TEXT    | Nullable; groups turns by conversation             |
| `role`      | TEXT    | `user` / `assistant` / `note` / `system`           |
| `content`   | TEXT    | The remembered text                                |
| `tags`      | TEXT    | Comma-separated, optional                          |
| `embedding` | BLOB    | Optional vector (only filled if embedder enabled)  |
| `ts`        | INTEGER | Unix epoch seconds                                 |

Indexed on `ts` and `(session_id, ts)`.

### `sessions`
Optional metadata. Kept thin on purpose — `turns.session_id` is the source of truth.

### `bloom_meta`
Single-row table holding `schema_version`. Bumping `SCHEMA_VERSION` in `db.py` and
adding a branch to `_migrate` is the only supported way to evolve the schema.

## Recall scoring

Default path (no embeddings):

1. Extract keywords from the query (alphanumeric tokens ≥3 chars, drop stopwords,
   pin ALL_CAPS / snake_case identifiers because they're high-signal).
2. SQLite `LIKE` search across `content` for any keyword. Pull `top_k * 3`
   candidates ordered by recency.
3. Score each candidate:
   - `+1.0` per keyword hit
   - `+2.0` if the turn is from the same session as the query
   - `+max(0, 2.0 - age_hours * 0.1)` — recency boost that decays over 20 hours
4. Sort descending, return top-k.

This is the same algorithm Bloom has used since v3, ported because it works and
needs no infrastructure.

## Embedders (optional)

Embedders are pluggable. Each implements:

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Embedders are imported lazily so the base install stays dependency-light.
You only pull in `openai`, `voyageai`, or `sentence-transformers` if you opt in.

When v0.2 lands, the `embedding` BLOB on `turns` will be filled at write time and
hybrid scoring (keyword + cosine similarity) will replace pure keyword scoring.

## Why SQLite

- One file. Trivial backup, trivial sync, trivial inspection (`sqlite3 ~/.bloom/loom.db`).
- No daemon, no port, no Docker.
- Plenty fast for personal use (millions of turns before write contention matters).
- WAL mode enabled, so concurrent reads don't block writes.

A Postgres backend is on the roadmap for team deployments. The `db.py` interface is
small enough that swapping it out is a weekend job.
