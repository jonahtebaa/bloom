"""MCP server — exposes Bloom's tools over stdio for Claude Code, Cursor,
Continue.dev, and any other MCP-compatible client.

Run with `bloom-mcp serve` (recommended) or `python -m bloom serve`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as mcp_types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from bloom import __version__
from bloom.config import Config
from bloom.db import Database
from bloom.tools import (
    tool_forget,
    tool_recall,
    tool_recent,
    tool_remember,
    tool_sessions,
    tool_stats,
)

log = logging.getLogger("bloom.server")


def _tool_definitions() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="recall",
            description=(
                "Search Bloom memory for past turns relevant to a query. "
                "Uses keyword + recency scoring; returns top-k matches with "
                "relevance scores. Use whenever the user references prior work, "
                "earlier conversations, or 'what did we say about X'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "default": 5, "description": "Max results"},
                    "session_filter": {
                        "type": "string",
                        "description": "Optional session id to bias toward",
                    },
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="remember",
            description=(
                "Persist a turn to Bloom memory so future `recall` calls can "
                "surface it. Use after meaningful decisions, learnings, or "
                "session-end summaries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text to remember"},
                    "session": {"type": "string", "description": "Session id"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "role": {
                        "type": "string",
                        "default": "note",
                        "description": "user | assistant | note | system",
                    },
                },
                "required": ["content"],
            },
        ),
        mcp_types.Tool(
            name="recent",
            description=(
                "Return the last N turns from a specific session, in chronological "
                "order. Good for resuming a paused session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "n": {"type": "integer", "default": 20},
                },
                "required": ["session_id"],
            },
        ),
        mcp_types.Tool(
            name="sessions",
            description="List known sessions with turn counts and timestamps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 50},
                },
            },
        ),
        mcp_types.Tool(
            name="forget",
            description="Delete a single turn by id.",
            inputSchema={
                "type": "object",
                "properties": {"turn_id": {"type": "integer"}},
                "required": ["turn_id"],
            },
        ),
        mcp_types.Tool(
            name="stats",
            description="Return DB stats: turn count, session count, size, schema version.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def build_server(cfg: Config | None = None) -> tuple[Server, Database, Config]:
    """Wire up the Server with handlers bound to a Database + Config."""
    cfg = cfg or Config.load()
    db = Database(cfg.db_path)
    server: Server = Server("bloom-mcp", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        try:
            if name == "recall":
                result = tool_recall(db, cfg, **arguments)
            elif name == "remember":
                result = tool_remember(db, cfg, **arguments)
            elif name == "recent":
                result = tool_recent(db, cfg, **arguments)
            elif name == "sessions":
                result = tool_sessions(db, cfg, **arguments)
            elif name == "forget":
                result = tool_forget(db, cfg, **arguments)
            elif name == "stats":
                result = tool_stats(db, cfg)
            else:
                result = {"ok": False, "error": f"unknown tool: {name}"}
        except TypeError as e:
            result = {"ok": False, "error": f"bad arguments: {e}"}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            result = {"ok": False, "error": str(e)}
        return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]

    return server, db, cfg


async def run_stdio() -> None:
    """Entry point for stdio transport — what `bloom-mcp serve` calls."""
    server, _db, cfg = build_server()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("bloom-mcp %s starting (db=%s, embedder=%s)",
             __version__, cfg.db_path, cfg.embedder.provider)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="bloom-mcp",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
