"""Bloom CLI — entry point for `bloom-mcp` (and `python -m bloom`).

Subcommands:
  init       Interactive first-time setup wizard.
  serve      Run the MCP server over stdio (what Claude Code invokes).
  stats      Print DB stats.
  register   Print or run the `claude mcp add` command for this install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bloom import __version__
from bloom.config import Config, EmbedderConfig, default_config_path, default_db_path, default_home


def _prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"{question}{suffix}: ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip()
    return answer or (default or "")


def _prompt_choice(question: str, choices: list[str], default: str) -> str:
    while True:
        joined = " / ".join(f"[{c}]" if c == default else c for c in choices)
        ans = _prompt(f"{question} ({joined})", default).lower()
        if ans in choices:
            return ans
        print(f"  please choose one of: {', '.join(choices)}")


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    ans = _prompt(f"{question} [{d}]").lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def _print_banner() -> None:
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print(f"  │  Bloom v{__version__}  —  persistent memory for       │")
    print("  │            Claude Code & MCP clients        │")
    print("  └─────────────────────────────────────────────┘")
    print()


def cmd_init(args: argparse.Namespace) -> int:
    """Interactive first-run setup."""
    _print_banner()
    home = default_home()
    print(f"Bloom home: {home}")
    print()

    if default_config_path().exists() and not args.force:
        print(f"Config already exists at {default_config_path()}")
        if not _prompt_yes_no("Overwrite?", default=False):
            print("Aborted. Use `bloom-mcp init --force` to skip this prompt.")
            return 0
    home.mkdir(parents=True, exist_ok=True)

    print("Step 1 — storage")
    print("  Bloom uses a single SQLite file. Default location is fine for most people.")
    db_path_str = _prompt("  Database path", default=str(default_db_path()))
    db_path = Path(db_path_str).expanduser()

    print()
    print("Step 2 — embedder (optional)")
    print("  Bloom's default `none` embedder uses keyword + recency scoring.")
    print("  It works fully offline with zero API keys and is what most users want.")
    print("  Pick a paid/local embedder only if you specifically want semantic search.")
    print()
    print("    none       — keyword scoring, no API key (default, recommended)")
    print("    openai     — OpenAI embeddings (requires OPENAI_API_KEY)")
    print("    anthropic  — Voyage AI (Anthropic's recommended; requires VOYAGE_API_KEY)")
    print("    local      — sentence-transformers, fully offline (~80 MB download)")
    provider = _prompt_choice(
        "  Embedder",
        ["none", "openai", "anthropic", "local"],
        default="none",
    )

    api_key_env: str | None = None
    model: str | None = None
    if provider == "openai":
        model = _prompt("  Model", default="text-embedding-3-small")
        api_key_env = "OPENAI_API_KEY"
        if not os.environ.get(api_key_env) and _prompt_yes_no(
            f"  {api_key_env} is not set. Enter it now? (will be added to ~/.bloom/.env)",
            default=True,
        ):
            key = _prompt(f"  {api_key_env}").strip()
            if key:
                _write_env(home / ".env", api_key_env, key)
                print(f"  Wrote {api_key_env} to {home / '.env'}")
    elif provider == "anthropic":
        model = _prompt("  Voyage model", default="voyage-3-lite")
        api_key_env = "VOYAGE_API_KEY"
        if not os.environ.get(api_key_env) and _prompt_yes_no(
            f"  {api_key_env} is not set. Enter it now? (will be added to ~/.bloom/.env)",
            default=True,
        ):
            key = _prompt(f"  {api_key_env}").strip()
            if key:
                _write_env(home / ".env", api_key_env, key)
                print(f"  Wrote {api_key_env} to {home / '.env'}")
    elif provider == "local":
        model = _prompt("  Sentence-transformers model", default="all-MiniLM-L6-v2")

    print()
    print("Step 3 — recall tuning (sensible defaults shown)")
    top_k = int(_prompt("  Default top_k for recall", default="5") or "5")
    max_chars = int(_prompt("  Max chars per recall response", default="4000") or "4000")

    cfg = Config(
        db_path=db_path,
        embedder=EmbedderConfig(provider=provider, model=model, api_key_env=api_key_env),
        retrieve_top_k=top_k,
        retrieve_max_chars=max_chars,
    )
    cfg_path = cfg.write()
    print()
    print(f"  ✓ Config written to {cfg_path}")

    print()
    print("Step 4 — Claude Code integration")
    if shutil.which("claude") and _prompt_yes_no(
        "  Register Bloom with Claude Code now?",
        default=True,
    ):
        _register_claude_code()
    else:
        print("  Skipped. Run `bloom-mcp register` later, or add manually:")
        print('    claude mcp add bloom -- bloom-mcp serve')

    print()
    print("Done. Try it:")
    print("    bloom-mcp stats")
    print("    bloom-mcp serve     # (Claude Code calls this for you)")
    print()
    return 0


def _write_env(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    lines = [ln for ln in existing.splitlines() if not ln.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def _register_claude_code() -> bool:
    cmd = ["claude", "mcp", "add", "bloom", "--", "bloom-mcp", "serve"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ✗ Could not run `claude mcp add`: {e}")
        return False
    if result.returncode == 0:
        print("  ✓ Registered with Claude Code as `bloom`.")
        return True
    print(f"  ✗ `claude mcp add` failed: {result.stderr.strip() or result.stdout.strip()}")
    return False


def cmd_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    from bloom.server import run_stdio

    asyncio.run(run_stdio())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:  # noqa: ARG001
    cfg = Config.load()
    from bloom.db import Database

    db = Database(cfg.db_path)
    s = db.stats()
    s["embedder"] = cfg.embedder.provider
    print(json.dumps(s, indent=2))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    if args.print_only:
        print("claude mcp add bloom -- bloom-mcp serve")
        return 0
    return 0 if _register_claude_code() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bloom-mcp",
        description="Persistent cross-session memory for Claude Code and MCP clients.",
    )
    parser.add_argument("--version", action="version", version=f"bloom-mcp {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Interactive first-time setup.")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config.")
    p_init.set_defaults(func=cmd_init)

    p_serve = sub.add_parser("serve", help="Run the MCP server (stdio).")
    p_serve.set_defaults(func=cmd_serve)

    p_stats = sub.add_parser("stats", help="Show DB stats.")
    p_stats.set_defaults(func=cmd_stats)

    p_reg = sub.add_parser("register", help="Register Bloom with Claude Code.")
    p_reg.add_argument(
        "--print-only",
        action="store_true",
        help="Print the command instead of running it.",
    )
    p_reg.set_defaults(func=cmd_register)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
