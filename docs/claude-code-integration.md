# Claude Code integration

Once Bloom is registered, Claude Code sees six new tools every session:

| Tool       | What it does                                                       |
|------------|--------------------------------------------------------------------|
| `recall`   | Search past turns by query — top-k by keyword + recency.           |
| `remember` | Persist a turn so future `recall`s can find it.                    |
| `recent`   | Last N turns of a specific session.                                |
| `sessions` | List known sessions (id, started_at, turn_count).                  |
| `forget`   | Delete a single turn by id.                                        |
| `stats`    | DB stats: size, schema version, embedder, total turns.             |

## Registering Bloom

### Automatic (recommended)

```bash
bloom-mcp init     # The wizard offers to do this for you.
```

### Manual

```bash
claude mcp add bloom -- bloom-mcp serve
```

Verify:

```bash
claude mcp list
```

You should see `bloom` in the output.

## Tip — capture session ends automatically

Claude Code supports `SessionStart` and `SessionEnd` hooks. A common pattern is
to have Claude write a one-line intent brief on session end, so future `recall`s
can find what each session was about.

Save as `~/.claude/hooks/bloom-session-end.sh`:

```bash
#!/usr/bin/env bash
# Append a session-end summary to Bloom.
# Claude Code passes hook context via env vars.
SUMMARY="${HOOK_SUMMARY:-Session ended at $(date -Iseconds)}"
SESSION_ID="${HOOK_SESSION_ID:-$(date +%s)}"

bloom-mcp << JSON
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"remember","arguments":{"content":"$SUMMARY","session":"$SESSION_ID","tags":"session-end"}}}
JSON
```

Then in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": "~/.claude/hooks/bloom-session-end.sh"
  }
}
```

The exact env var names depend on your Claude Code version — `claude --help` shows
current hook semantics. (The simpler approach is just to instruct Claude in your
`CLAUDE.md` to call `remember` itself before ending major sessions.)

## How Claude knows when to call Bloom

Bloom's tool descriptions are written so Claude triggers them naturally:

- `recall` description includes "Use whenever the user references prior work,
  earlier conversations, or 'what did we say about X'."
- `remember` includes "Use after meaningful decisions, learnings, or
  session-end summaries."

If you want stronger habits (e.g. always recall on session start), drop a line
in your global `CLAUDE.md`:

```markdown
- Always run `recall` at the start of any session where the user references
  past work, and `remember` after major decisions.
```

## Troubleshooting

**`bloom` doesn't appear in `claude mcp list`** — Run `bloom-mcp register` again,
or check `claude mcp logs bloom` for errors. Most common cause: `bloom-mcp` is not
on your PATH (pipx installs to `~/.local/bin`, which needs to be on PATH).

**Claude calls `recall` but returns no results** — Check `bloom-mcp stats`. If
`turns: 0`, you haven't `remember`-ed anything yet. Bloom doesn't auto-capture by
default; that's a session-end hook concern.

**Permission errors on `~/.bloom/loom.db`** — Bloom creates its home with default
permissions. If you ran `bloom-mcp init` as root and now run as a user, just
`chown -R you:you ~/.bloom`.
