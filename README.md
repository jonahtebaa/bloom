# Bloom

**Persistent cross-session memory for [Claude Code](https://claude.com/claude-code) and any MCP-compatible client. Built to survive Opus 4.7's small context window.**

Bloom is a single-file SQLite memory that survives between sessions. Ask Claude Code "what did we decide about X last week" and it can actually answer — by searching every prior turn it has saved.

---

## Why Bloom — the small-context problem

If you've used Claude Opus 4.7 in long sessions, you already know the pain:

- **Auto-compaction kicks in early.** 4.7's working context is tighter than 4.6's, so the model starts losing detail mid-session.
- **Cross-session memory is zero.** Open a new session tomorrow and Claude has no idea what you decided today.
- **`--resume` is one session.** It doesn't help when you're switching between projects, days, or machines.

Bloom is the direct counter to that:

1. **Recall on demand, not always-on.** `recall("the postgres decision")` pulls the exact original turn back into context only when needed — ~200 tokens to retrieve, vs. 20K to keep it loaded the whole session.
2. **Session-end checkpointing.** `remember` the *conclusion* of a working session, throw away the verbose path. The next session resumes with the result, not the journey.
3. **SessionStart hook = automatic warm-up.** Every new session opens with a short "recent memory" block injected at startup, so 4.7 doesn't begin empty.
4. **Forever continuity.** 4.7's compaction is per-session and ephemeral; Bloom is persistent. Day 1's "we picked X" is queryable on day 30.

The trick is keeping recall **out** of the model's context until it's needed. Loading 200 turns at session start would defeat the purpose. Loading the 5 most relevant on demand is the architecture.

---

```
$ pipx install bloom-mcp
$ bloom-mcp init
$ # Bloom is now wired into Claude Code. Open a new session and ask it to recall.
```

- **Works offline.** Default scoring is keyword + recency, backed by SQLite FTS5 — no API key, no network.
- **Embedder hooks ready, not yet active.** OpenAI / Voyage / local adapters
  are wired but unused in v0.2 — recall today is keyword + recency. Semantic
  recall lands in v0.3 (see roadmap).
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
- **`recall`** — search by query; get the top-k most relevant past turns ranked by keyword overlap (SQLite FTS5), recency, and — once v0.3 lands — semantic similarity.
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

> **Status (v0.2):** embedder support is wired but not yet active. Recall in
> v0.2 uses keyword + recency scoring backed by SQLite FTS5. The adapters
> below load and validate cleanly today; semantic search across stored
> embeddings is the v0.3 milestone — see the roadmap.

Bloom's default `none` embedder uses keyword + recency scoring. It's fast, offline, and good enough for most use cases.

When v0.3 ships, you'll be able to pick one of:

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

All values can also be set via env vars: `BLOOM_DB_PATH`, `BLOOM_EMBEDDER`,
`BLOOM_EMBEDDER_MODEL`, `BLOOM_RETRIEVE_TOP_K`, `BLOOM_LOG_LEVEL`,
`BLOOM_HOME`. Any keys saved to `~/.bloom/.env` (e.g. by the `init` wizard)
are auto-loaded into the environment on startup.

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

- [x] v0.1 — initial SQLite-backed recall, six MCP tools, init wizard
- [x] v0.2 — FTS5 keyword search, soft-delete, SessionStart auto-recall hook
- [ ] v0.3 — embedding-augmented recall (hybrid keyword + cosine)
- [ ] v0.4 — multi-tenant team mode (shared host, per-user namespaces)
- [ ] v0.5 — pruning + compaction (auto-summarize old turns to save space)
- [ ] v0.6 — alternative backends (Postgres, DuckDB)

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
