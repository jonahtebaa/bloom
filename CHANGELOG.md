# Changelog

All notable changes to Bloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- numpy is now a declared runtime dependency (was implicit).
- Hybrid recall no longer spoofs BM25 scores for pure-semantic candidates
  (rows that match by cosine but share no keyword with the query). The
  BM25 contribution now reflects the true FTS5 rank for keyword candidates
  and is `0.0` for pure-semantic candidates.
- `tool_stats` now exposes both the legacy `turns`/`sessions` keys and the
  more descriptive `total_turns`/`total_sessions` aliases so callers built
  against either name work.
- `tool_remember` parameter renamed `session` -> `session_id` to align with
  the MCP server's tool schema.
- Semantic recall pool now uses a stratified sample (recent + older random)
  instead of just the 200 most-recent embedded turns, so old keyword-miss
  turns can still surface via cosine similarity on large DBs.

### Added
- **Embedders are now active.** When `embedder` is set to `openai`,
  `anthropic` (Voyage AI), or `local` in `~/.bloom/config.toml`, `remember`
  computes a document embedding at write-time and stores it in the row's
  `embedding` BLOB; `recall` searches BOTH the FTS5 keyword candidates
  AND a semantic pool (most recent N rows with embeddings, default 200)
  and scores the union with `0.4*bm25 + 0.5*cosine + 0.1*recency`.
  Failures (missing optional dep, missing API key, network error,
  malformed blob) fall back to keyword-only ranking and never crash
  the call.
- `bloom-mcp backfill-embeddings` — iterates rows where `embedding IS
  NULL` and writes embeddings in batches. Handles SIGINT cleanly.
- `bloom-mcp doctor` — prints diagnostics (Bloom version, config path,
  DB path + schema version, embedder, turn/session counts) plus the
  contents of `~/.bloom/last_hook_error.log` if a previous SessionStart
  hook invocation failed silently.
- `bloom-mcp purge --hard` — permanently removes soft-deleted rows.
  `--hard` is required so it never runs by accident.
- New embedder protocol surface: `embed_doc(content)` and `embed_query(query)`
  return numpy float32 arrays; `dim = 0` is the no-op sentinel.
- `db.update_embedding`, `db.fetch_embeddings`, `db.iter_missing_embeddings`,
  `db.count_missing_embeddings`, and `db.fetch_recent_with_embeddings`
  helpers (db.py stays numpy-free).
- FTS5 search backend for content recall (replaces naive `LIKE` scan).
- `filter_session` and `session_bias` as distinct arguments to `recall` —
  hard-filter to one session vs. soft-prefer it.
- Content size cap (256 KB per turn) and bounds-checking on `k`, `n`, and
  `limit` arguments across all tools.
- `BLOOM_EMBEDDER_MODEL`, `BLOOM_RETRIEVE_TOP_K`, and `BLOOM_DEBUG`
  environment variables.

### Changed
- `forget` is now a soft-delete (sets `deleted_at`) so accidental deletes
  are recoverable until you run `bloom-mcp purge --hard`. Schema
  bumped to v3.
- SessionStart hook dedup uses a structured `_bloom_marker` JSON field rather
  than a shell-comment marker glued onto the command string. Old installs are
  upgraded automatically when `install-hook` is re-run.
- `recall-print` now actually surfaces recalled turns (seeded by cwd basename
  + git branch), falling back to the recent-sessions metadata block only when
  no keyword hits are found. Matches what the README advertises.
- Hook installer writes `settings.json` atomically (`os.replace`) and refuses
  to clobber a non-list `SessionStart` value rather than silently rewriting it.
- The wizard's "Register with Claude Code" step now defaults to **no** —
  some `claude` builds emit a Bun crash trace from non-interactive
  invocations and that scared users mid-wizard. Run `bloom-mcp register`
  later if you want it.
- `bloom-mcp register` / wizard registration wraps the `claude mcp add`
  subprocess so a non-zero exit (or runtime crash) collapses to a single
  actionable line plus the manual command, instead of blowing up the wizard.

### Fixed
- v1 → v3 migration now actually rebuilds the FTS index — the previous
  `INSERT ... NOT IN turns_fts` guard silently skipped pre-existing rows
  on external-content FTS5, leaving them unsearchable forever.
- `recall(filter_session=...)` is now a true hard filter: the
  `session_id` constraint is pushed into the SQL WHERE so the LIMIT
  applies after filtering. Previously, N strong cross-session matches
  could displace the only matching row in the requested session.
- Semantic recall is now retrieval, not just reranking: the candidate
  pool is the union of FTS5 hits AND a recent-rows-with-embeddings pool,
  so a keyword-miss query can surface a semantically-relevant turn.
- Threading lock around DB writes — `Database` now serialises every
  `execute` with an `RLock`. Concurrent writers were getting their
  cursors interleaved and writes silently dropped.
- Code tokens (`c++`, `c#`, `f#`, `.NET`, `k8s`, `i18n`, `l10n`) are
  preserved in keyword extraction. The generic `\w+` tokenizer was
  shredding them into single letters / `net`.
- Cloud embedder dimensions are now probed from the actual API
  response on first use, rather than hard-coded — so model swaps
  (`text-embedding-3-small` → `…-large`) work without a code edit.
- `~/.bloom/.env` is now a secrets-only sidecar: only credential-shaped
  keys (`*_API_KEY`, `*_TOKEN`, `*_SECRET`, plus the known auth env
  names) are loaded into `os.environ`. Any `BLOOM_*` line is ignored
  with a one-line stderr warning so a stale `.env` can't silently
  override `config.toml`. Resolution order is now: env vars >
  config.toml > built-in defaults; `.env` only supplies secrets.
- `bloom-mcp backfill-embeddings` requires `--confirm` for cloud
  embedders (openai/anthropic) — backfill is a bulk-send and users
  shouldn't be able to upload their full memory store by reflex.
- `recall-print` ALWAYS writes a one-line trace to
  `~/.bloom/last_hook_error.log` on failure (truncate, not append) so
  silent SessionStart-hook failures are visible via `bloom-mcp doctor`.
  The hook still returns 0 so a broken Bloom can't break a Claude Code
  session; set `BLOOM_DEBUG=1` to additionally re-raise.
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
