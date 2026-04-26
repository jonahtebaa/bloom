# Configuration

Bloom reads configuration from three places (highest precedence first):

1. Environment variables.
2. `~/.bloom/config.toml` (or `$BLOOM_HOME/config.toml`).
3. Built-in defaults.

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

[logging]
level = "INFO"           # DEBUG | INFO | WARNING | ERROR
```

## Environment variables

Useful for CI, containers, or one-off shell overrides.

| Variable           | Purpose                                          |
|--------------------|--------------------------------------------------|
| `BLOOM_HOME`       | Override the `~/.bloom` directory entirely.      |
| `BLOOM_DB_PATH`    | Override the SQLite path.                        |
| `BLOOM_EMBEDDER`   | `none` / `openai` / `anthropic` / `local`.       |
| `BLOOM_LOG_LEVEL`  | Logging level for the MCP server.                |
| `OPENAI_API_KEY`   | Required if `embedder.provider = "openai"`.      |
| `VOYAGE_API_KEY`   | Required if `embedder.provider = "anthropic"`.   |

You can also drop a `.env` file at `~/.bloom/.env`; the install wizard writes
API keys there with `chmod 600` permissions.

## Choosing an embedder

| Provider      | When to use                                                    |
|---------------|----------------------------------------------------------------|
| `none`        | Default. Works offline. Right answer for ~90% of users.        |
| `openai`      | You already have an OpenAI key and want best-in-class semantic. |
| `anthropic`   | You're on the Anthropic stack and want Voyage embeddings.      |
| `local`       | You want semantic search but no external API. ~80MB model download. |

Embeddings only matter when keywords miss — e.g. you remembered "we picked the
queue" and later search "what database tech did we go with for the work pipeline".
Pure keyword recall would miss that. Embeddings would catch it.

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
