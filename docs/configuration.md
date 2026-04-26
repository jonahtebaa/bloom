# Configuration

Bloom reads configuration from three places (highest precedence first):

1. Environment variables.
2. `~/.bloom/config.toml` (or `$BLOOM_HOME/config.toml`).
3. Built-in defaults.

`~/.bloom/.env` is a **secrets-only** sidecar — it is NOT a config-override
mechanism. Bloom reads it after `config.toml` and only honours
credential-shaped keys (e.g. `OPENAI_API_KEY`, `VOYAGE_API_KEY`,
`ANTHROPIC_API_KEY`, anything ending `_API_KEY` / `_TOKEN` / `_SECRET`).
Any `BLOOM_*` line in `.env` is ignored with a one-line stderr warning so
config can't be silently overridden by a stale `.env` file. Put your
Bloom config in `config.toml` and your secrets in `.env` (or your shell
environment).

The `bloom-mcp init` wizard writes a sane `config.toml` for you. Most users
never touch it.

## File: `~/.bloom/config.toml`

```toml
[storage]
db_path = "/home/you/.bloom/loom.db"   # Anywhere you want; created on first use.

[recall]
top_k = 5                # Default number of results returned by `recall`.
max_chars = 4000         # Cap on total snippet text per response.
snippet_max_chars = 600  # Cap per individual turn before truncation.

[embedder]
provider = "none"        # none | openai | anthropic | local
# model = "text-embedding-3-small"   # Optional, provider-specific.
# api_key_env = "OPENAI_API_KEY"     # Which env var holds the key.

[semantic]
pool_size = 200          # Max recent rows considered for cosine re-rank
                         # (only used when an embedder is configured).

[logging]
level = "INFO"           # DEBUG | INFO | WARNING | ERROR
```

## Environment variables

Useful for CI, containers, or one-off shell overrides.

| Variable                | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `BLOOM_HOME`            | Override the `~/.bloom` directory entirely.                          |
| `BLOOM_DB_PATH`         | Override the SQLite path.                                            |
| `BLOOM_EMBEDDER`        | `none` / `openai` / `anthropic` / `local`.                           |
| `BLOOM_EMBEDDER_MODEL`  | Provider-specific model name (e.g. `text-embedding-3-small`).        |
| `BLOOM_RETRIEVE_TOP_K`  | Default `top_k` for recall (integer).                                |
| `BLOOM_LOG_LEVEL`       | Logging level for the MCP server (`DEBUG` / `INFO` / `WARNING` / `ERROR`). |
| `BLOOM_DEBUG`           | `1` = re-raise exceptions inside the `recall-print` SessionStart hook (defaults to silent + log file). |
| `OPENAI_API_KEY`        | Required if `embedder.provider = "openai"`.                          |
| `VOYAGE_API_KEY`        | Required if `embedder.provider = "anthropic"`.                       |

You can also drop a `.env` file at `~/.bloom/.env`. The install wizard
writes API keys there with `chmod 600` permissions. **`.env` is for
secrets only** — `BLOOM_*` keys in `.env` are ignored with a stderr
warning. Put your Bloom config in `config.toml`.

## Choosing an embedder

| Provider      | When to use                                                    |
|---------------|----------------------------------------------------------------|
| `none`        | Default. Works offline. Right answer for ~90% of users.        |
| `openai`      | You already have an OpenAI key and want best-in-class semantic. |
| `anthropic`   | You're on the Anthropic stack and want Voyage embeddings.      |
| `local`       | You want semantic search but no external API. ~80MB model download. |

Embeddings only matter when keywords miss — e.g. you remembered "we picked
the queue" and later search "what database tech did we go with for the
work pipeline". Pure keyword recall would miss that. Embeddings catch it
because Bloom's hybrid recall searches BOTH pools when an embedder is
configured: the FTS5 keyword candidates AND a semantic pool of the most
recent rows that have an embedding (`[semantic] pool_size = 200` by
default), unioned and scored together. A query that shares no keyword
with the stored turn can still surface — it just needs cosine ≥ 0.2.

## Storage tuning

If your `loom.db` grows large (say 100K+ turns):

- It's still fine. SQLite handles millions of rows happily.
- If recall feels slow, consider running `VACUUM` periodically.
- Compaction (auto-summarize old turns into summaries) lands in v0.4.

## Multiple Bloom installs

Run more than one Bloom (e.g. one personal, one per-project) by setting
`BLOOM_HOME` differently per shell:

```bash
BLOOM_HOME=~/projects/myapp/.bloom bloom-mcp serve
```

Then register each one with Claude Code under a unique name:

```bash
claude mcp add bloom-myapp -- env BLOOM_HOME=~/projects/myapp/.bloom bloom-mcp serve
```
