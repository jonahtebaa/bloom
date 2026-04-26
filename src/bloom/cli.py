"""Bloom CLI — entry point for `bloom-mcp` (and `python -m bloom`).

Subcommands:
  init           Interactive first-time setup wizard.
  serve          Run the MCP server over stdio (what Claude Code invokes).
  stats          Print DB stats.
  register       Print or run the `claude mcp add` command for this install.
  install-hook   Install Claude Code SessionStart hook so Bloom auto-recalls
                 prior context at the start of every session.
  recall-print   Print a recent-memory block to stdout (used by the hook).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bloom import __version__
from bloom.config import (
    Config,
    ConfigError,
    EmbedderConfig,
    default_config_path,
    default_db_path,
    default_home,
)


# --- Prompts -----------------------------------------------------------------

_SECRET_TOKENS = ("secret", "password", "passphrase")


def _is_secret_name(name: str) -> bool:
    n = name.lower()
    if n.endswith("_api_key") or n.endswith("api_key"):
        return True
    return any(tok in n for tok in _SECRET_TOKENS)


def _prompt(question: str, default: str | None = None) -> str:
    """Prompt the user. If the question name looks like a secret, hide input."""
    suffix = f" [{default}]" if default else ""
    line = f"{question}{suffix}: "
    if _is_secret_name(question):
        try:
            answer = getpass.getpass(line)
        except (EOFError, KeyboardInterrupt):
            return default or ""
        return answer.strip() or (default or "")
    sys.stdout.write(line)
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


# --- Banner ------------------------------------------------------------------

_BANNER_BOX = (
    "  ┌─────────────────────────────────────────────┐",
    "  │  Bloom v{ver}  —  persistent memory for       │",
    "  │            Claude Code & MCP clients        │",
    "  └─────────────────────────────────────────────┘",
)
_BANNER_ASCII = (
    "  +---------------------------------------------+",
    "  |  Bloom v{ver}  --  persistent memory for      |",
    "  |            Claude Code & MCP clients        |",
    "  +---------------------------------------------+",
)


def _stdout_supports_unicode() -> bool:
    enc = (sys.stdout.encoding or "").lower().replace("-", "")
    return enc in ("utf8",)


def _print_banner() -> None:
    lines = _BANNER_BOX if _stdout_supports_unicode() else _BANNER_ASCII
    print()
    for ln in lines:
        formatted = ln.format(ver=__version__)
        try:
            print(formatted)
        except UnicodeEncodeError:
            # Last-ditch fallback if the terminal lies about its encoding.
            print(_BANNER_ASCII[lines.index(ln)].format(ver=__version__))
    print()


# --- Wizard helpers (one per step) ------------------------------------------


def _step_db_path() -> Path:
    print("Step 1 — storage")
    print("  Bloom uses a single SQLite file. Default location is fine for most people.")
    db_path_str = _prompt("  Database path", default=str(default_db_path()))
    return Path(db_path_str).expanduser()


def _step_embedder(home: Path) -> EmbedderConfig:
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
        _maybe_collect_api_key(api_key_env, home)
    elif provider == "anthropic":
        model = _prompt("  Voyage model", default="voyage-3-lite")
        api_key_env = "VOYAGE_API_KEY"
        _maybe_collect_api_key(api_key_env, home)
    elif provider == "local":
        model = _prompt("  Sentence-transformers model", default="all-MiniLM-L6-v2")

    return EmbedderConfig(provider=provider, model=model, api_key_env=api_key_env)


def _maybe_collect_api_key(api_key_env: str, home: Path) -> None:
    if os.environ.get(api_key_env):
        return
    if not _prompt_yes_no(
        f"  {api_key_env} is not set. Enter it now? (will be added to ~/.bloom/.env)",
        default=True,
    ):
        return
    key = _prompt(f"  {api_key_env}").strip()
    if key:
        _write_env(home / ".env", api_key_env, key)
        print(f"  Wrote {api_key_env} to {home / '.env'}")


def _step_register() -> None:
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


def _step_install_hook() -> None:
    print()
    print("Step 5 — Auto-recall on every Claude Code session (optional)")
    print("  This installs a SessionStart hook so Claude Code automatically")
    print("  loads recent Bloom memories at the start of every session.")
    print("  Without this, Claude only uses Bloom when you explicitly ask.")
    if _prompt_yes_no("  Install the SessionStart auto-recall hook?", default=True):
        ok, msg = install_session_start_hook()
        print(f"  {'OK' if ok else 'X'} {msg}")
    else:
        print("  Skipped. Run `bloom-mcp install-hook` later if you change your mind.")


# --- cmd_init ---------------------------------------------------------------


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

    db_path = _step_db_path()
    embedder = _step_embedder(home)

    print()
    print("Step 3 — recall tuning (sensible defaults shown)")
    top_k = int(_prompt("  Default top_k for recall", default="5") or "5")
    max_chars = int(_prompt("  Max chars per recall response", default="4000") or "4000")

    cfg = Config(
        db_path=db_path,
        embedder=embedder,
        retrieve_top_k=top_k,
        retrieve_max_chars=max_chars,
    )
    cfg_path = cfg.write()
    print()
    print(f"  OK Config written to {cfg_path}")

    _step_register()
    _step_install_hook()

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
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
    except FileNotFoundError:
        print(
            "  X `claude` CLI not found on PATH. Install Claude Code first, "
            "then run `bloom-mcp register`."
        )
        return False
    except subprocess.TimeoutExpired:
        print("  X `claude mcp add` timed out after 10s.")
        return False
    if result.returncode == 0:
        print("  OK Registered with Claude Code as `bloom`.")
        return True
    print(f"  X `claude mcp add` failed: {result.stderr.strip() or result.stdout.strip()}")
    return False


# --- serve / stats / register ------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    try:
        cfg = Config.load()  # noqa: F841 — load to surface ConfigError early
    except ConfigError as e:
        print(f"bloom-mcp: configuration error: {e}", file=sys.stderr)
        return 2

    from bloom.server import run_stdio

    asyncio.run(run_stdio())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:  # noqa: ARG001
    try:
        cfg = Config.load()
    except ConfigError as e:
        print(f"bloom-mcp: configuration error: {e}", file=sys.stderr)
        return 2
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


# --- SessionStart hook -------------------------------------------------------

# Legacy shell-comment marker (still recognised on read so older installs are
# rewritten cleanly), plus the new structured marker stored as a JSON field on
# the hook entry itself — argparse-safe and easy to dedupe on.
SESSION_START_HOOK_MARKER = "# bloom-mcp:session-start"
BLOOM_MARKER_FIELD = "_bloom_marker"
BLOOM_MARKER_VALUE = "session-start"


def _is_bloom_entry(entry: Any) -> bool:
    """Return True if `entry` looks like a Bloom-installed SessionStart hook.

    Recognises both the new structured `_bloom_marker` field and the legacy
    shell-comment marker on the inner command, so re-running the installer
    cleanly upgrades an old entry.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get(BLOOM_MARKER_FIELD) == BLOOM_MARKER_VALUE:
        return True
    for h in entry.get("hooks") or []:
        if isinstance(h, dict) and SESSION_START_HOOK_MARKER in (h.get("command") or ""):
            return True
    return False


def _atomic_write_settings(settings_path: Path, payload: str) -> None:
    """Write `payload` to `settings_path` atomically, preserving perms."""
    tmp = settings_path.with_suffix(".json.tmp")
    tmp.write_text(payload)
    if settings_path.exists():
        try:
            shutil.copymode(settings_path, tmp)
        except OSError:
            pass
    os.replace(tmp, settings_path)


def install_session_start_hook(
    settings_path: Path | None = None,
    n_recent: int = 5,
) -> tuple[bool, str]:
    """Wire a SessionStart hook into ~/.claude/settings.json so Claude Code
    runs `bloom-mcp recall-print` at the start of every session.

    Idempotent: re-running replaces the prior Bloom hook entry without
    touching the user's other hooks. Refuses to clobber an existing
    SessionStart that isn't a list (the documented Claude Code shape).
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
    existing_session_start = hooks.get("SessionStart")

    if existing_session_start is not None and not isinstance(existing_session_start, list):
        return (
            False,
            "existing SessionStart is not a list — refusing to overwrite. "
            f"Edit {settings_path} manually and re-run.",
        )

    bloom_cmd = f"bloom-mcp recall-print --k {n_recent}"
    new_entry = {
        BLOOM_MARKER_FIELD: BLOOM_MARKER_VALUE,
        "matcher": "*",
        "hooks": [{"type": "command", "command": bloom_cmd}],
    }

    session_start = list(existing_session_start or [])
    session_start = [e for e in session_start if not _is_bloom_entry(e)]
    session_start.append(new_entry)
    hooks["SessionStart"] = session_start

    _atomic_write_settings(settings_path, json.dumps(settings, indent=2) + "\n")
    return True, f"hook installed in {settings_path}"


def cmd_install_hook(args: argparse.Namespace) -> int:
    ok, msg = install_session_start_hook(n_recent=args.n)
    print(f"{'OK' if ok else 'X'} {msg}")
    return 0 if ok else 1


# --- recall-print ------------------------------------------------------------


def _git_branch() -> str:
    """Best-effort current branch name for the cwd, or empty on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if result.returncode != 0:
        return ""
    branch = result.stdout.strip()
    return "" if branch in ("HEAD", "") else branch


def _format_recall_block(scored: list[Any], snippet_chars: int = 240) -> list[str]:
    lines = ["===== BLOOM MEMORY | recalled turns ====="]
    for t in scored:
        ts = int(getattr(t, "ts", 0) or 0)
        when = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if ts
            else "—"
        )
        role = (getattr(t, "role", None) or "?")[:8]
        content = (getattr(t, "content", "") or "").strip().replace("\n", " ")
        if len(content) > snippet_chars:
            content = content[: snippet_chars - 1] + "…"
        lines.append(f"  - [{when}] ({role}) {content}")
    lines.append("")
    lines.append(
        "Bloom MCP tools available: recall(query, k), remember(content, session, tags),"
    )
    lines.append("recent(session_id, n), sessions(), forget(turn_id), stats().")
    lines.append("Use `recall` whenever the user references prior work.")
    lines.append("==========================================")
    return lines


def cmd_recall_print(args: argparse.Namespace) -> int:
    """Print recalled Bloom memories as plaintext — invoked by the SessionStart hook.

    We seed recall with `<cwd basename> <git branch>` so the warm-up block is
    project-relevant, not just chronological. Failures are silent so a broken
    Bloom never breaks a Claude Code session — set `BLOOM_DEBUG=1` to surface
    the underlying exception while developing.
    """
    try:
        cfg = Config.load()
        from bloom.db import Database
        from bloom.recall import recall as recall_fn

        db = Database(cfg.db_path)

        cwd_name = Path.cwd().name
        branch = _git_branch()
        seed = f"{cwd_name} {branch}".strip()

        scored = recall_fn(db, seed, k=args.k) if seed else []
        if scored:
            print("\n".join(_format_recall_block(scored)))
            return 0

        # No keyword hits yet — fall back to the recent-sessions metadata block
        # so the user still gets *something* useful at session start.
        sessions = db.list_sessions(limit=args.k)
        if not sessions:
            return 0
        lines = ["===== BLOOM MEMORY | recent sessions ====="]
        for s in sessions:
            ts = int(s["last_ts"] or 0)
            when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"  - [{when}] session={s['id']} ({s['turn_count']} turns)")
        lines.append("")
        lines.append("Use `recall(query)` to search across all stored turns.")
        lines.append("==========================================")
        print("\n".join(lines))
        return 0
    except Exception:  # noqa: BLE001
        if os.environ.get("BLOOM_DEBUG"):
            raise
        return 0


# --- argparse / main ---------------------------------------------------------


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
        help="Print recalled memory summary to stdout (used by the SessionStart hook).",
    )
    p_print.add_argument("--k", type=int, default=5, help="Number of turns to surface.")
    p_print.set_defaults(func=cmd_recall_print)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
