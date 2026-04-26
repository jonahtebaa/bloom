"""SessionStart hook installer + recall-print formatter."""

from __future__ import annotations

import json
from pathlib import Path

from bloom.cli import SESSION_START_HOOK_MARKER, install_session_start_hook


def test_install_hook_creates_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    ok, _ = install_session_start_hook(settings_path=settings)
    assert ok
    data = json.loads(settings.read_text())
    entries = data["hooks"]["SessionStart"]
    assert any(
        SESSION_START_HOOK_MARKER in h["command"]
        for e in entries
        for h in e["hooks"]
    )


def test_install_hook_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_session_start_hook(settings_path=settings)
    install_session_start_hook(settings_path=settings)
    data = json.loads(settings.read_text())
    bloom_entries = [
        e
        for e in data["hooks"]["SessionStart"]
        if any(SESSION_START_HOOK_MARKER in h.get("command", "") for h in e["hooks"])
    ]
    assert len(bloom_entries) == 1


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
    assert any(SESSION_START_HOOK_MARKER in c for c in cmds)


def test_install_hook_rejects_broken_json(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{ this is not json")
    ok, msg = install_session_start_hook(settings_path=settings)
    assert ok is False
    assert "parse" in msg.lower()
