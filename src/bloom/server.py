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
from bloom.embedders import load_embedder
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
            description="Search Bloom memory for past turns matching a query, ranked by keyword overlap and recency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Max results",
                    },
                    "session_bias": {
                        "type": "string",
                        "description": "Optional session id to boost (results from this session rank higher)",
                    },
                    "filter_session": {
                        "type": "string",
                        "description": "Optional session id to restrict results to (hard filter)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="remember",
            description="Persist a single turn to Bloom memory so future recall calls can surface it.",
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
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="recent",
            description="Return the last N turns from a specific session, in chronological order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "n": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="sessions",
            description="List known sessions with turn counts and timestamps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="forget",
            description="Soft-delete a single turn by id so it no longer appears in recall or recent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "turn_id": {"type": "integer", "minimum": 1},
                },
                "required": ["turn_id"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="stats",
            description="Return DB stats: turn count, session count, size, schema version.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]


def build_server(cfg: Config | None = None) -> tuple[Server, Database, Config]:
    """Wire up the Server with handlers bound to a Database + Config.

    Loads the embedder ONCE at startup. If load_embedder fails (missing
    optional dep, missing API key, init crash) it transparently returns the
    no-op embedder, so a misconfigured embedder never prevents startup.
    """
    cfg = cfg or Config.load()
    db = Database(cfg.db_path)
    embedder = load_embedder(cfg.embedder)
    server: Server = Server("bloom-mcp", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        try:
            if name == "recall":
                result = tool_recall(db, cfg, embedder=embedder, **arguments)
            elif name == "remember":
                result = tool_remember(db, cfg, embedder=embedder, **arguments)
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
    log.info(
        "bloom-mcp %s starting (db=%s, embedder=%s)",
        __version__,
        cfg.db_path,
        cfg.embedder.provider,
    )

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
