# Onboarding — getting Bloom set up in under 2 minutes

Bloom gives Claude Code a memory that survives between sessions. Once you've
installed it, you can ask Claude things like *"what did we decide about X last
week?"* and it can actually answer — by searching every prior turn it has saved
to your local memory.

This guide gets you from zero to a fully-wired install. **Everything stays on
your machine.** No cloud, no telemetry, no shared backend.

---

## Step 1 — install Bloom

Pick the install method that matches your setup. **`pipx` is recommended** — it
isolates Bloom from your system Python so it can never conflict with other
tools.

### Option A — `pipx` (recommended)

```bash
# Install pipx itself if you don't already have it:
#   macOS:    brew install pipx && pipx ensurepath
#   Ubuntu:   sudo apt install pipx && pipx ensurepath
#   Windows:  python -m pip install --user pipx && python -m pipx ensurepath

pipx install bloom-mcp
```

### Option B — `uv`

```bash
uv tool install bloom-mcp
```

### Option C — Homebrew (macOS / Linuxbrew)

```bash
brew install jonahtebaa/bloom/bloom-mcp
```

### Option D — plain `pip`

```bash
pip install --user bloom-mcp
```

Verify the install:

```bash
bloom-mcp --version
# bloom-mcp 0.1.0
```

If you see `command not found`, your `PATH` is missing `~/.local/bin`. Run
`pipx ensurepath` (or `uv tool update-shell`), then open a new terminal.

---

## Step 2 — run the setup wizard

```bash
bloom-mcp init
```

It walks you through five steps. **You can press Enter on every single prompt
and the defaults will work.** The full sequence:

1. **Storage** — where to put your SQLite memory file. Default `~/.bloom/loom.db`
   is right for almost everyone.
2. **Embedder** — leave on `none` unless you specifically know you want
   semantic search. The default is fast, free, offline, and good enough for
   most use cases.
3. **Recall tuning** — accept the defaults.
4. **Claude Code integration** — say yes. The wizard runs
   `claude mcp add bloom -- bloom-mcp serve` for you.
5. **SessionStart hook** — say yes. This is the magic that makes Bloom
   *automatic*. (See "How auto-recall works" below.)

When it's done, open a new Claude Code session and you should see something
like this near the top:

```
===== BLOOM MEMORY | recent sessions =====
  - [2026-04-26 09:42 UTC] session=cc-a993f71b (16 turns)
  - [2026-04-26 06:14 UTC] session=cc-a341b709 (3 turns)

Bloom MCP tools available: recall(query, k), remember(content, session, tags), ...
==========================================
```

That's Bloom telling Claude what it has on file. From here, every meaningful
turn you `remember` becomes searchable in every future session.

---

## Step 3 — use it

There's nothing more to do. Claude Code now has six new tools:

| Tool       | What it does                                                       |
|------------|--------------------------------------------------------------------|
| `recall`   | Search past turns by query — top-k by keyword + recency.           |
| `remember` | Persist a turn so future sessions can find it.                     |
| `recent`   | Last N turns of a specific session.                                |
| `sessions` | List known sessions.                                               |
| `forget`   | Delete a single turn by id.                                        |
| `stats`    | DB stats.                                                          |

Just talk to Claude normally. Try things like:

- *"Search Bloom for what we discussed about the database migration."*
- *"Save to Bloom: client X wants the proposal by Friday."*
- *"What did we decide about the API rate limiting yesterday?"*

Claude will pick the right tool on its own.

---

## How auto-recall works

When you accept the SessionStart hook in step 2, the wizard adds a small entry
to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          { "type": "command", "command": "bloom-mcp recall-print --k 5 # bloom-mcp:session-start" }
        ]
      }
    ]
  }
}
```

Every time Claude Code starts a session, it runs `bloom-mcp recall-print`,
which prints a short summary of recent Bloom sessions to stdout. Claude Code
includes that output in the conversation context so the assistant knows what
you've been working on. This is exactly the "autostarts every session" behavior
that the project author uses personally.

If you skipped the hook in the wizard, install it any time:

```bash
bloom-mcp install-hook
```

To uninstall: edit `~/.claude/settings.json` and delete the entry whose
command contains `# bloom-mcp:session-start`.

---

## Tips for getting the most out of Bloom

**Tell Claude to remember things explicitly.** *"Save to Bloom: we decided to
use Postgres for the queue."* Claude will call `remember` with that content
and tag it appropriately.

**Use sessions to group related work.** When you're starting a long project,
say *"start a Bloom session called proj-alpha"*. Then when you return:
*"recall recent turns from proj-alpha"*.

**Add it to your CLAUDE.md.** A line like this trains the model to use Bloom
proactively:

```markdown
- Always run `recall` at the start of any session where I reference past work.
- Always `remember` the outcome of major decisions and end-of-session summaries.
```

**Inspect your memory directly.** Bloom is just SQLite — you can poke around:

```bash
sqlite3 ~/.bloom/loom.db 'SELECT id, role, content FROM turns ORDER BY ts DESC LIMIT 10'
```

Or use the CLI:

```bash
bloom-mcp stats           # how much have I remembered?
```

---

## Troubleshooting

**`bloom` doesn't appear in `claude mcp list`** — Run `bloom-mcp register`
again, then restart Claude Code.

**Claude doesn't seem to use Bloom** — Add it to your `CLAUDE.md` so the model
knows to. Also check `bloom-mcp stats` — if `turns: 0`, you haven't told it to
remember anything yet.

**The auto-recall summary doesn't appear at session start** — Check
`~/.claude/settings.json` actually contains the bloom entry. Run
`bloom-mcp install-hook` to (re-)add it.

**Permission errors on `~/.bloom/loom.db`** — Most often happens if you ran
`bloom-mcp init` as `sudo` and now try as your regular user.
Fix: `sudo chown -R $USER:$USER ~/.bloom`.

**I want to wipe my memory and start fresh** —
`rm ~/.bloom/loom.db && bloom-mcp stats` (it'll re-create an empty one).

---

## What about teammates?

Right now Bloom is **per-user, local-first**. Each install gets its own SQLite
file. There's no built-in shared/team mode — that's coming in v0.3.

If you want to share decisions or context with a colleague today, just paste
what `bloom-mcp recall <query>` returns into Slack or your team chat.

---

## Where to get help

- **Docs**: [github.com/jonahtebaa/bloom](https://github.com/jonahtebaa/bloom) — README, configuration, security
- **Issues**: [github.com/jonahtebaa/bloom/issues](https://github.com/jonahtebaa/bloom/issues)
- **Internal Webspot**: ping Jonah directly
