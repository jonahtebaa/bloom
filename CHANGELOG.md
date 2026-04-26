# Changelog

All notable changes to Bloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Embedders are now active.** When `embedder` is set to `openai`,
  `anthropic` (Voyage AI), or `local` in `~/.bloom/config.toml`, `remember`
  computes a document embedding at write-time and stores it in the row's
  `embedding` BLOB; `recall` embeds the query once and re-ranks the top
  FTS5 candidates with `0.4*bm25 + 0.5*cosine + 0.1*recency`. Failures
  (missing optional dep, missing API key, network error, malformed blob)
  fall back to keyword-only ranking and never crash the call.
- `bloom-mcp backfill-embeddings` — iterates rows where `embedding IS
  NULL` and writes embeddings in batches. Handles SIGINT cleanly.
- New embedder protocol surface: `embed_doc(content)` and `embed_query(query)`
  return numpy float32 arrays; `dim = 0` is the no-op sentinel.
- `db.update_embedding`, `db.fetch_embeddings`, `db.iter_missing_embeddings`,
  and `db.count_missing_embeddings` helpers (db.py stays numpy-free).
- FTS5 search backend for content recall (replaces naive `LIKE` scan).
- `filter_session` and `session_bias` as distinct arguments to `recall` —
  hard-filter to one session vs. soft-prefer it.
- Content size cap (256 KB per turn) and bounds-checking on `k`, `n`, and
  `limit` arguments across all tools.
- `BLOOM_EMBEDDER_MODEL` and `BLOOM_RETRIEVE_TOP_K` environment variables.
- Auto-loading of `~/.bloom/.env` on `Config.load()` so saved API keys reach
  the embedders without an extra shell-export step.

### Changed
- `forget` is now a soft-delete (sets `deleted_at`) so accidental deletes
  are recoverable. Schema bumped to v3.
- SessionStart hook dedup uses a structured `_bloom_marker` JSON field rather
  than a shell-comment marker glued onto the command string. Old installs are
  upgraded automatically when `install-hook` is re-run.
- `recall-print` now actually surfaces recalled turns (seeded by cwd basename
  + git branch), falling back to the recent-sessions metadata block only when
  no keyword hits are found. Matches what the README advertises.
- Hook installer writes `settings.json` atomically (`os.replace`) and refuses
  to clobber a non-list `SessionStart` value rather than silently rewriting it.

### Fixed
- `fetch_recent` returned the oldest N turns instead of the newest N.
- Keyword extractor now handles unicode and CJK input correctly (previously
  ASCII-only word boundaries dropped non-Latin tokens entirely).
- API keys typed into the `init` wizard no longer echo to the terminal —
  the prompt uses `getpass.getpass()` for any `_API_KEY` / secret field.
- Banner falls back to ASCII when stdout is not UTF-8 (prevents
  `UnicodeEncodeError` on locked-down Windows code pages).
- Malformed `~/.bloom/config.toml` now raises a friendly `ConfigError` with
  the file path and underlying parse error instead of a bare traceback.
- Wizard no longer crashes the whole flow when the `claude` CLI is absent —
  prints a clean message and continues.

## [0.2.0] - 2026-04-26

### Added
- `bloom-mcp install-hook` — installs a Claude Code SessionStart hook so Bloom
  auto-fires at the start of every session, surfacing recent memory context to
  the assistant. The wizard now offers this as Step 5.
- `bloom-mcp recall-print` — prints recent-session summary to stdout in a
  Claude-friendly format. Invoked by the SessionStart hook.
- Documentation: `docs/onboarding.md` — a 2-minute employee onboarding guide
  covering install methods, the wizard, auto-recall, usage tips, and
  troubleshooting.

### Changed
- Wizard now has five steps instead of four (added auto-recall hook).

## [0.1.0] - 2026-04-26

### Added
- Initial release.
- SQLite-backed cross-session memory for Claude Code (and any MCP-compatible client).
- MCP server with six tools: `recall`, `remember`, `recent`, `sessions`, `forget`, `stats`.
- Keyword + recency recall scoring — works fully offline, no API keys required.
- Embedder plugin interface with `none` (default), `openai`, `anthropic`, `local` adapters.
- Interactive `bloom-mcp init` wizard for first-time setup.
- TOML configuration at `~/.bloom/config.toml`.
- CLI subcommands: `init`, `serve`, `stats`, `register` (Claude Code MCP registration).
