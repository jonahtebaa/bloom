# Security & data handling

## What Bloom stores

Whatever you ask it to. Bloom never auto-captures conversations. Every turn in
`~/.bloom/loom.db` got there because:

1. You (or Claude, on your behalf) called `remember`.
2. A hook script you set up appended it.

There is no telemetry. There is no cloud component. There is no auto-upload.

## Where Bloom stores it

By default: `~/.bloom/loom.db` (SQLite). The directory is created with default
filesystem permissions (your user, mode 0755 for the dir, 0644 for the DB on
most Linux/macOS setups). The DB itself does not encrypt-at-rest.

If your threat model includes "another local user reads my memory":

```bash
chmod 700 ~/.bloom
chmod 600 ~/.bloom/loom.db
```

If your threat model includes "stolen laptop":

- Use full-disk encryption (FileVault / LUKS / BitLocker).
- Or store the DB on an encrypted volume and point `BLOOM_DB_PATH` at it.

## API keys

If you choose an `openai`, `anthropic`, or other paid embedder, Bloom needs an
API key. The install wizard writes it to `~/.bloom/.env` with mode 0600 (user
read/write only). You can also set the key via your shell environment, in which
case Bloom never persists it.

### Outbound data flow per embedder choice

By default, Bloom is **offline**: keyword + recency FTS5 ranking, zero
network. The `none` embedder is the recommended default for exactly this
reason.

When you enable an embedder, each `remember` call sends the new turn's
content to the configured embedding provider, and each `recall` sends
the query string. Specifically:

- `local` (sentence-transformers) — runs on-device. The model is
  downloaded once on first use; nothing leaves the machine after that.
- `openai` — turn content and query text are sent over HTTPS to OpenAI's
  embeddings endpoint. Subject to OpenAI's data-handling policy.
- `anthropic` (Voyage AI) — turn content and query text are sent over
  HTTPS to Voyage's endpoint. Subject to Voyage's data-handling policy.

`bloom-mcp backfill-embeddings` is the one **bulk** operation Bloom
performs: it sends every existing live turn (without an embedding) to
the configured embedder. That can be hundreds or thousands of rows in
one go. Run it only when you trust the provider with your full memory
store. As a safety rail, the command **requires `--confirm`** for cloud
embedders (`openai`, `anthropic`) — without it, the command refuses and
prints what it would have sent. The `local` embedder doesn't need
`--confirm` because nothing leaves the machine.

If you want zero outbound network traffic, use `embedder.provider = "none"`
(the default) or `local`.

## Multi-user systems

Bloom is designed for one user per install. On a shared host, run separate
Bloom instances per user with distinct `BLOOM_HOME` values. Don't share a
`loom.db` between users — there's no access control inside the DB.

## Reporting security issues

If you find a vulnerability, **please don't open a public GitHub issue**.
Instead, email the maintainer (see project metadata) with:

- A description of the issue.
- Steps to reproduce.
- The affected version.

You'll get a response within 7 days.

## Threat model summary

| Threat                                          | Bloom protects? | How                                       |
|-------------------------------------------------|-----------------|-------------------------------------------|
| Network eavesdropper sees my memory             | Yes             | Default install has no network component. |
| Embedder API provider sees my queries           | Partially       | Only if you opt into a paid embedder.     |
| Another local user reads my `loom.db`           | No (default)    | Use `chmod 700 ~/.bloom`.                 |
| Stolen laptop with no FDE                       | No              | Use full-disk encryption.                 |
| Malicious dependency (supply chain)             | Partially       | Pinned `mcp` version; minimal deps.       |
| Bloom server runs untrusted code from a client  | N/A             | MCP tools only read/write text + ints.    |
