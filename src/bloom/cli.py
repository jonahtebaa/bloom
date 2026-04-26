"""Bloom CLI — entry point for `bloom-mcp` (and `python -m bloom`).

Subcommands:
  init           Interactive first-time setup wizard.
  serve          Run the MCP server over stdio (what Claude Code invokes).
  stats          Print DB stats.
  register       Print or run the `claude mcp add` command for this install.
  install-hook   Install Claude Code SessionStart hook so Bloom auto-recalls
                 prior context at the start of every session.
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
from typing import Any

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
    print("Step 5 — Auto-recall on every Claude Code session (optional)")
    print("  This installs a SessionStart hook so Claude Code automatically")
    print("  loads recent Bloom memories at the start of every session.")
    print("  Without this, Claude only uses Bloom when you explicitly ask.")
    if _prompt_yes_no("  Install the SessionStart auto-recall hook?", default=True):
        ok, msg = install_session_start_hook()
        print(f"  {'✓' if ok else '✗'} {msg}")
    else:
        print("  Skipped. Run `bloom-mcp install-hook` later if you change your mind.")

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


SESSION_START_HOOK_MARKER = "# bloom-mcp:session-start"


def install_session_start_hook(
    settings_path: Path | None = None,
    n_recent: int = 5,
) -> tuple[bool, str]:
    """Wire a SessionStart hook into ~/.claude/settings.json so Claude Code
    runs `bloom-mcp recent-print` at the start of every session.

    Idempotent: re-running replaces the prior Bloom hook entry without
    touching the user's other hooks.
    """
    settings_path = settings_path or (Path.home() / ".claude" / "settings.json")
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text() or "{}")
        except json.JSONDecodeError:
            return False, f"could not parse {settings_path} — fix the JSON and retry"

    hooks = settings.setdefault("hooks", {})
    session_start = hooks.setdefault("SessionStart", [])

    bloom_cmd = f"bloom-mcp recall-print --k {n_recent} {SESSION_START_HOOK_MARKER}"
    new_entry = {
        "matcher": "*",
        "hooks": [{"type": "command", "command": bloom_cmd}],
    }

    if isinstance(session_start, list):
        session_start = [
            e for e in session_start
            if not (
                isinstance(e, dict)
                and any(
                    SESSION_START_HOOK_MARKER in (h.get("command") or "")
                    for h in (e.get("hooks") or [])
                    if isinstance(h, dict)
                )
            )
        ]
        session_start.append(new_entry)
        hooks["SessionStart"] = session_start
    else:
        hooks["SessionStart"] = [new_entry]

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return True, f"hook installed in {settings_path}"


def cmd_install_hook(args: argparse.Namespace) -> int:
    ok, msg = install_session_start_hook(n_recent=args.n)
    print(f"{'✓' if ok else '✗'} {msg}")
    return 0 if ok else 1


def cmd_recall_print(args: argparse.Namespace) -> int:
    """Print recent Bloom memories as plaintext — invoked by the SessionStart hook.

    Output goes to stdout in a Claude-friendly format. Failures are silent
    (we never want to break a session because Bloom can't read its DB).
    """
    try:
        cfg = Config.load()
        from bloom.db import Database

        db = Database(cfg.db_path)
        sessions = db.list_sessions(limit=args.k)
        if not sessions:
            return 0
        lines = ["===== BLOOM MEMORY | recent sessions ====="]
        for s in sessions:
            ts = int(s["last_ts"] or 0)
            from datetime import datetime, timezone

            when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(
                f"  - [{when}] session={s['id']} ({s['turn_count']} turns)"
            )
        lines.append("")
        lines.append(
            "Bloom MCP tools available: recall(query, k), remember(content, session, tags),"
        )
        lines.append("recent(session_id, n), sessions(), forget(turn_id), stats().")
        lines.append("Use `recall` whenever the user references prior work.")
        lines.append("==========================================")
        print("\n".join(lines))
        return 0
    except Exception:  # noqa: BLE001
        return 0


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

    p_hook = sub.add_parser(
        "install-hook",
        help="Install a Claude Code SessionStart hook so Bloom auto-fires every session.",
    )
    p_hook.add_argument("--n", type=int, default=5, help="Recent sessions to surface.")
    p_hook.set_defaults(func=cmd_install_hook)

    p_print = sub.add_parser(
        "recall-print",
        help="Print recent memory summary to stdout (used by the SessionStart hook).",
    )
    p_print.add_argument("--k", type=int, default=5, help="Number of recent sessions.")
    p_print.add_argument("marker", nargs="?", help="Internal marker; ignored.")
    p_print.set_defaults(func=cmd_recall_print)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
