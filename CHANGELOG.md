# Changelog

All notable changes to Bloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
