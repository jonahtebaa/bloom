# Contributing

Bloom is small on purpose. The bar for new features is high; the bar for bug
fixes and clarity improvements is low.

## Setup

```bash
git clone https://github.com/jonahtebaa/bloom.git
cd bloom
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Style

- `ruff check src tests` must pass.
- `mypy src` should pass; we tolerate but don't accumulate ignores.
- Type hints on all public functions.
- One purpose per module; if a file is doing two things, split it.
- No comments that explain *what* the code does — only *why*, and only when
  non-obvious.

## Tests

- Every behavior change ships with a test.
- Tests use a `tmp_path` fixture for the DB — never write to `~/.bloom`.
- Aim for fast tests (the full suite runs in well under a second).

## What's in scope

- Better recall scoring.
- New embedder providers (with optional extras).
- Schema migrations (carefully — bump `SCHEMA_VERSION` and add a `_migrate` branch).
- Better Claude Code integration helpers.
- Performance work on recall path.

## What's out of scope (for now)

- A built-in web UI. Bloom is a library + MCP server.
- Real-time sync between machines. (See "team mode" on the roadmap.)
- LLM-side summarization. Bloom stores what you give it; summarization belongs
  in the agent calling `remember`.

## Pull request flow

1. Open an issue first for non-trivial changes — gives space to align on
   approach before code.
2. Branch from `main`. Keep PRs small (< ~300 LOC where possible).
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Make sure CI is green.

## Releasing (maintainers only)

```bash
# Bump version in src/bloom/__init__.py and pyproject.toml.
# Move CHANGELOG entries from [Unreleased] to a new version block.
git tag v0.X.Y
git push --tags
python -m build
twine upload dist/bloom_mcp-0.X.Y*
```
