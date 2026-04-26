# Changelog

All notable changes to Bloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
