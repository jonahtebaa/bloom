"""SessionStart hook installer + recall-print formatter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

from bloom.cli import (
    BLOOM_MARKER_FIELD,
    BLOOM_MARKER_VALUE,
    SESSION_START_HOOK_MARKER,
    cmd_recall_print,
    install_session_start_hook,
)


def _bloom_entries(data: dict) -> list[dict]:
    return [
        e
        for e in data["hooks"]["SessionStart"]
        if isinstance(e, dict) and e.get(BLOOM_MARKER_FIELD) == BLOOM_MARKER_VALUE
    ]


def test_install_hook_creates_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    ok, _ = install_session_start_hook(settings_path=settings)
    assert ok
    data = json.loads(settings.read_text())
    entries = data["hooks"]["SessionStart"]
    assert len(_bloom_entries(data)) == 1
    # New shape: dedup marker is a JSON field, NOT a shell comment in the cmd.
    cmds = [h["command"] for e in entries for h in e["hooks"]]
    assert any("bloom-mcp recall-print" in c for c in cmds)
    assert all(SESSION_START_HOOK_MARKER not in c for c in cmds)
    assert all("#" not in c for c in cmds)


def test_install_hook_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_session_start_hook(settings_path=settings)
    install_session_start_hook(settings_path=settings)
    data = json.loads(settings.read_text())
    assert len(_bloom_entries(data)) == 1


def test_install_hook_preserves_other_hooks(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "*",
                            "hooks": [{"type": "command", "command": "echo other"}],
                        }
                    ]
                }
            }
        )
    )
    install_session_start_hook(settings_path=settings)
    data = json.loads(settings.read_text())
    cmds = [h["command"] for e in data["hooks"]["SessionStart"] for h in e["hooks"]]
    assert "echo other" in cmds
    assert any("bloom-mcp recall-print" in c for c in cmds)
    assert len(_bloom_entries(data)) == 1


def test_install_hook_rejects_broken_json(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{ this is not json")
    ok, msg = install_session_start_hook(settings_path=settings)
    assert ok is False
    assert "parse" in msg.lower()


def test_install_hook_refuses_non_list_session_start(tmp_path: Path) -> None:
    """If somebody hand-edited SessionStart into a dict, don't clobber it."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": {
                        "matcher": "*",
                        "command": "echo legacy single-entry",
                    }
                }
            }
        )
    )
    ok, msg = install_session_start_hook(settings_path=settings)
    assert ok is False
    assert "not a list" in msg.lower()
    # Original content untouched.
    data = json.loads(settings.read_text())
    assert isinstance(data["hooks"]["SessionStart"], dict)


def test_install_hook_uses_atomic_replace(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    with mock.patch("bloom.cli.os.replace", wraps=__import__("os").replace) as spy:
        ok, _ = install_session_start_hook(settings_path=settings)
    assert ok
    assert spy.called, "expected os.replace to be used for atomic settings.json write"
    args, _ = spy.call_args
    src, dst = args
    assert str(dst) == str(settings)
    assert str(src).endswith(".json.tmp")


def test_recall_print_writes_error_log_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """A broken Config.load inside cmd_recall_print → log file is written, exit 0."""
    bloom_home = tmp_path / "bloom-home"
    bloom_home.mkdir()
    monkeypatch.setenv("BLOOM_HOME", str(bloom_home))
    monkeypatch.delenv("BLOOM_DEBUG", raising=False)

    log = bloom_home / "last_hook_error.log"
    assert not log.exists()

    def boom(*_a, **_kw):
        raise RuntimeError("simulated config explosion")

    monkeypatch.setattr("bloom.cli.Config.load", boom)

    args = argparse.Namespace(k=5)
    rc = cmd_recall_print(args)

    # Hook MUST never break the session, regardless of what blew up.
    assert rc == 0
    assert log.exists(), "expected hook error log to be written on failure"
    text = log.read_text()
    assert "RuntimeError" in text
    assert "simulated config explosion" in text


def test_install_hook_upgrades_legacy_marker(tmp_path: Path) -> None:
    """An entry from the old shell-comment installer should be replaced cleanly."""
    settings = tmp_path / "settings.json"
    legacy_cmd = f"bloom-mcp recall-print --k 5 {SESSION_START_HOOK_MARKER}"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "*",
                            "hooks": [{"type": "command", "command": legacy_cmd}],
                        }
                    ]
                }
            }
        )
    )
    install_session_start_hook(settings_path=settings)
    data = json.loads(settings.read_text())
    bloom = _bloom_entries(data)
    assert len(bloom) == 1
    # Legacy entry was removed, not duplicated.
    all_cmds = [h["command"] for e in data["hooks"]["SessionStart"] for h in e["hooks"]]
    assert legacy_cmd not in all_cmds
