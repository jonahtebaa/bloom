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

### Default path (no embeddings)

1. Extract keywords from the query — `\w+` tokens of length ≥2 (plus a
   whitelist of code-shaped tokens like `c++`, `c#`, `f#`, `.NET`, `k8s`,
   `i18n`, `l10n` which the generic tokenizer would otherwise shred), with
   stopwords dropped, ALL_CAPS / SNAKE_CAPS identifiers pinned to the front
   (high signal), and continuous CJK runs treated as standalone tokens.
2. SQLite **FTS5** search over `turns_fts` (an external-content virtual
   table mirroring `turns.content`). Candidates come back ordered by
   `bm25(turns_fts)`, ascending. `filter_session` is pushed into the SQL
   WHERE clause so the LIMIT applies *after* the session filter.
3. Score each candidate:
   - `+1.0` per keyword hit
   - `+2.0` if the turn is from the same session as the query
   - `+2.0 * exp(-age_days / 14.0)` — exponential recency decay
     (half-life ≈ 9.7 days; older never collapses to zero, so yesterday
     and last-year don't tie)
4. Sort descending, return top-k.

### Hybrid path (embedder configured)

When an embedder is configured (`openai` / `anthropic` / `local`), recall
unions **two** candidate pools and re-scores the union:

- **Keyword pool** — the FTS5 candidates above.
- **Semantic pool** — the most recent N live rows that have an
  `embedding` BLOB stored (default `N = 200`, configurable via
  `[semantic] pool_size` in `config.toml` or the `semantic_pool_size`
  arg to `recall()`). Each row's stored embedding is loaded once,
  cosine'd against the query embedding, and the top-M survive.

The union is deduped by `id` and scored as
`0.4 * bm25_norm + 0.5 * cosine + 0.1 * recency_norm + 0.05 * same_session_bias`.
A keyword-miss row that doesn't share any token with the query can still
make it through — it just needs cosine ≥ 0.2.

If `embedder.embed_query` raises (network down, missing dep, bad key) we
log to stderr and fall back cleanly to keyword-only scoring. Recall
itself never raises.

## Embedders (optional)

Embedders are pluggable. Each implements the protocol in
`src/bloom/embedders/base.py`:

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed_doc(self, text: str) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    # Optional batch API for backfill:
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...
```

`dim = 0` is the no-op sentinel: any embedder that fails to initialise
(missing optional dep, bad API key) reports `dim = 0` and recall
transparently degrades to keyword-only ranking. Cloud providers
(`openai`, `anthropic`/Voyage) probe their actual response dimensionality
on first use rather than hard-coding it.

Embedders are imported lazily so the base install stays dependency-light.
You only pull in `openai`, `voyageai`, or `sentence-transformers` if you
opt in (`pip install bloom-mcp[openai]` etc.).

## Schema migrations

`bloom_meta.schema_version` tracks the current shape; `db.py::_migrate`
runs each missing branch in order:

| From → To | Change                                                            |
|-----------|-------------------------------------------------------------------|
| 0 → 1     | Initial `turns` and `sessions` tables, indexes on `ts`.            |
| 1 → 2     | Adds the `turns_fts` FTS5 external-content virtual table + sync   |
|           | triggers + a `'rebuild'` of the FTS index over pre-existing rows. |
| 2 → 3     | Adds `turns.deleted_at` (soft delete) + index, and reroutes       |
|           | `forget` from hard-delete to soft-delete.                         |

Bumping `SCHEMA_VERSION` and adding a new branch in `_migrate` is the
only supported way to evolve the schema — never drop columns in place.

## Threading model

`db.Database` keeps a single long-lived `sqlite3.Connection` and serializes
all access with a re-entrant `threading.RLock`. The connection opens with
`check_same_thread=False`, but the lock is mandatory: SQLite's connection
is not safe for interleaved `execute` calls across threads, and concurrent
writes were getting lost without it. Callers that nest `with db.connect():`
are fine because the lock is an `RLock`.

## Why SQLite

- One file. Trivial backup, trivial sync, trivial inspection (`sqlite3 ~/.bloom/loom.db`).
- No daemon, no port, no Docker.
- Plenty fast for personal use (millions of turns before write contention matters).
- WAL mode enabled, so concurrent reads don't block writes.

A Postgres backend is on the roadmap for team deployments. The `db.py` interface is
small enough that swapping it out is a weekend job.
