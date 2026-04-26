"""Threaded-write probe for the cached sqlite3 connection.

The Database holds a single sqlite3 connection with `check_same_thread=False`;
without serialization, two threads racing on `insert_turn` will interleave
prepared-statement state on the shared cursor and either lose writes or
raise InterfaceError/SystemError. The fix is a per-instance threading.RLock
held for the duration of every `connect()` block.

This test spawns 8 threads each calling `insert_turn` 50 times. Expectation:
exactly 400 rows inserted, no exceptions raised.
"""

from __future__ import annotations

import threading
from pathlib import Path

from bloom.db import Database


def test_concurrent_inserts_do_not_lose_writes(tmp_path: Path) -> None:
    db = Database(tmp_path / "loom.db")

    n_threads = 8
    per_thread = 50
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for i in range(per_thread):
                db.insert_turn(
                    content=f"thread-{tid} row-{i}",
                    session_id=f"s{tid}",
                    role="user",
                )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors!r}"
    stats = db.stats()
    assert stats["turns"] == n_threads * per_thread, (
        f"expected {n_threads * per_thread} rows, got {stats['turns']}"
    )


def test_concurrent_reads_and_writes_safe(tmp_path: Path) -> None:
    """A reader thread polling stats() while writers insert must not raise.

    sqlite3.Connection isn't thread-safe even for read-only access on a
    shared connection, so this exercises the same lock as the write probe.
    """
    db = Database(tmp_path / "loom.db")

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                db.stats()
                db.list_sessions(limit=10)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def writer(tid: int) -> None:
        try:
            for i in range(40):
                db.insert_turn(content=f"w{tid} r{i}", session_id=f"s{tid}")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    r = threading.Thread(target=reader, daemon=True)
    r.start()
    writers = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    r.join(timeout=2.0)

    assert not errors, f"errors during concurrent r/w: {errors!r}"
    assert db.stats()["turns"] == 4 * 40
