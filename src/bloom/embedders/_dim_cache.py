"""Persistent dimension cache for embedder providers.

OpenAI / Voyage adapters historically guessed `self.dim` from the model name
string (`"small" → 1536`, `"lite" → 512`). This is fragile: a new or custom
model name silently lands on the default branch and produces wrong-dim
vectors that break cosine math at recall time.

Instead, every adapter should set `self.dim` from a real probe — call the
provider once with a tiny input and read `len(vec)`. We cache the discovered
dim per `(provider, model)` in `~/.bloom/.embedder_cache.json` so subsequent
process startups don't re-probe (which would cost an API call every cold
start).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _cache_path() -> Path:
    home = os.environ.get("BLOOM_HOME") or str(Path.home() / ".bloom")
    return Path(home) / ".embedder_cache.json"


def _load() -> dict[str, Any]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — corrupt cache, treat as empty
        return {}


def _save(data: dict[str, Any]) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        # Best-effort cache; if the FS write fails (read-only mount, perms),
        # the next process will just re-probe.
        pass


def get_cached_dim(provider: str, model: str) -> int | None:
    data = _load()
    val = data.get(f"{provider}:{model}")
    if isinstance(val, int) and val > 0:
        return val
    return None


def set_cached_dim(provider: str, model: str, dim: int) -> None:
    if not (isinstance(dim, int) and dim > 0):
        return
    data = _load()
    data[f"{provider}:{model}"] = int(dim)
    _save(data)
