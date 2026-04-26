# Bloom

**Persistent cross-session memory for [Claude Code](https://claude.com/claude-code) and any MCP-compatible client.**

Bloom is a single-file SQLite memory that survives between sessions. Ask Claude Code "what did we decide about X last week" and it can actually answer — by searching every prior turn it has saved.

```
$ pipx install bloom-mcp
$ bloom-mcp init
$ # Bloom is now wired into Claude Code. Open a new session and ask it to recall.
```

- **Works offline.** Default scoring is keyword + recency — no API key, no network.
- **Optional embeddings.** OpenAI, Voyage (Anthropic-recommended), or local sentence-transformers.
- **Six MCP tools.** `recall`, `remember`, `recent`, `sessions`, `forget`, `stats`.
- **One file, one process.** SQLite at `~/.bloom/loom.db`. No daemon, no Docker, no cloud.
- **MIT-licensed.** Use it however you want.

> **Note** — Bloom is an independent open-source project, not affiliated with Anthropic.
> "Claude" and "Claude Code" are trademarks of Anthropic PBC.

---

## Quickstart

### 1. Install

The recommended way is [`pipx`](https://pypa.github.io/pipx/) (isolates Bloom from your system Python):

```bash
pipx install bloom-mcp
```

Or if you prefer plain `pip` / `uv`:

```bash
pip install bloom-mcp
# or
uv tool install bloom-mcp
```

### 2. Set up

```bash
bloom-mcp init
```

The wizard will:
1. Pick a database location (default: `~/.bloom/loom.db`).
2. Choose an embedder (`none` is recommended — works offline, no API key).
3. Tune recall settings.
4. Register Bloom with Claude Code automatically (if `claude` is on your PATH).

### 3. Use it

In Claude Code, the assistant now has six new tools available. Try:

> "Search Bloom for what we said about the postgres migration."

Claude will call `recall("postgres migration")` and surface relevant past turns.

---

## What Bloom does

**Bloom solves one problem: Claude Code forgets every session.** Even with `--resume`, you lose context across days/weeks/projects. Bloom adds a tiny memory layer:

- **`remember`** — store a turn (decision, learning, summary) so future sessions can find it.
- **`recall`** — search by query; get the top-k most relevant past turns ranked by keyword overlap, recency, and (optionally) semantic similarity.
- **`recent`** — pull the last N turns of a specific session.
- **`sessions`** — list known sessions and their turn counts.
- **`forget`** — delete a single turn by id.
- **`stats`** — DB size, schema version, embedder.

You can use it as a Python library too:

```python
from bloom.config import Config
from bloom.db import Database
from bloom.tools import tool_remember, tool_recall

cfg = Config.load()
db = Database(cfg.db_path)

tool_remember(db, cfg, content="we picked Postgres SKIP LOCKED for the queue", session="proj-x", tags="decision")
print(tool_recall(db, cfg, query="queue choice"))
```

---

## Embedders (optional)

Bloom's default `none` embedder uses keyword + recency scoring. It's fast, offline, and good enough for most use cases.

If you want semantic search ("find turns about *that thing we discussed*" without needing the exact words), pick one of:

| Provider | Install | Auth | Cost |
|---|---|---|---|
| `none` | (default) | — | Free |
| `openai` | `pip install bloom-mcp[openai]` | `OPENAI_API_KEY` | ~$0.02 / 1M tokens |
| `anthropic` (Voyage AI) | `pip install bloom-mcp[anthropic]` | `VOYAGE_API_KEY` | ~$0.02 / 1M tokens |
| `local` | `pip install bloom-mcp[local]` | — (downloads model) | Free, ~80 MB |

Set in `~/.bloom/config.toml` or via `bloom-mcp init`:

```toml
[embedder]
provider = "openai"
model = "text-embedding-3-small"
```

---

## Configuration

`~/.bloom/config.toml`:

```toml
[storage]
db_path = "/home/you/.bloom/loom.db"

[recall]
top_k = 5
max_chars = 4000
snippet_max_chars = 600

[embedder]
provider = "none"

[logging]
level = "INFO"
```

All values can also be set via env vars: `BLOOM_DB_PATH`, `BLOOM_EMBEDDER`, `BLOOM_LOG_LEVEL`.

---

## Sharing Bloom with a team

Bloom is **per-user, local-first by design**. Every install gets its own SQLite file. There is no cloud component, no telemetry, no shared backend. Your conversations live on your machine.

If you want a shared team memory, the right approach is:

1. Each person installs Bloom locally (they get private memory).
2. Run a second Bloom instance on a shared host with a team-only namespace.
3. Use a small wrapper (not yet shipped — track [issue #1](https://github.com/jonahtebaa/bloom/issues/1)) that fans out `remember` to both.

A first-party "team Bloom" mode is on the roadmap but not in v0.1.

---

## How it compares

- **vs [mem0](https://github.com/mem0ai/mem0)** — Bloom is local-first, single-file, MCP-native. Mem0 is hosted/cloud-first with richer extraction.
- **vs raw vector DBs (Qdrant, pgvector)** — Bloom is *the application*, not the storage layer. You get tools, scoring, CLI, and Claude Code integration out of the box.
- **vs Claude Code's `--resume`** — Resume is one session. Bloom is *every* session, searchable.

---

## Roadmap

- [ ] v0.2 — embedding-augmented recall (hybrid keyword + cosine)
- [ ] v0.3 — multi-tenant team mode (shared host, per-user namespaces)
- [ ] v0.4 — pruning + compaction (auto-summarize old turns to save space)
- [ ] v0.5 — alternative backends (Postgres, DuckDB)

---

## Documentation

- **[Onboarding](docs/onboarding.md)** — 2-minute setup guide for new users
- [Architecture](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Claude Code integration](docs/claude-code-integration.md)
- [Security & data handling](docs/security.md)
- [Contributing](docs/contributing.md)

---

## License

[MIT](LICENSE) — © 2026 Jonah Tebaa.
