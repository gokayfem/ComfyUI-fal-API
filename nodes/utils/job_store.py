"""Persistent inbox of async fal jobs so they survive ComfyUI restarts.

Every job queued through Fal Submit is recorded in a ``jobs`` table inside
the same sqlite database as the result cache ("submit tonight, collect
tomorrow"). When a result is later fetched by request id — even in a fresh
session — the job is marked collected.

Every public method is best-effort: any sqlite failure degrades to a no-op
or an empty result — bookkeeping must never break generation.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any

from .logger import logger
from .result_cache import _default_db_path

_PRUNE_AFTER_DAYS = 30
_SECONDS_PER_DAY = 86400.0

_STATUS_SUBMITTED = "submitted"
_STATUS_COLLECTED = "collected"

_COLUMNS = ("request_id", "endpoint", "status", "submitted_at", "collected_at", "note")

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        request_id TEXT PRIMARY KEY,
        endpoint TEXT,
        status TEXT,
        submitted_at REAL,
        collected_at REAL,
        note TEXT
    )
"""


def _humanize_age(seconds: float) -> str:
    """Compact age like '45s', '12m', '2h' or '3d'. Clamped at zero."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < _SECONDS_PER_DAY:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // _SECONDS_PER_DAY)}d"


def _format_entry(entry: dict[str, Any], now: float) -> str:
    """One report line: '  ⏳ 2h ago  fal-ai/kling-video/v3  req=abc123'."""
    collected = entry.get("status") == _STATUS_COLLECTED
    icon = "✅" if collected else "⏳"
    reference = entry.get("collected_at") if collected else entry.get("submitted_at")
    if not isinstance(reference, (int, float)):
        reference = entry.get("submitted_at")
    age = (
        f"{_humanize_age(now - float(reference))} ago"
        if isinstance(reference, (int, float))
        else "age unknown"
    )
    endpoint = entry.get("endpoint") or "(unknown endpoint)"
    return f"  {icon} {age}  {endpoint}  req={entry.get('request_id') or '-'}"


class JobStore:
    """Thread-safe singleton over the persistent async-job inbox."""

    _instance: JobStore | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> JobStore:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._connect_failed = False
        self._db_path = _default_db_path()

    # -- connection -----------------------------------------------------------

    def _connection(self) -> sqlite3.Connection | None:
        """Open (once) and return the sqlite connection; None if unavailable.

        Must be called with ``self._lock`` held. A corrupted or unwritable
        database disables the job store for the session instead of raising.
        """
        if self._conn is not None:
            return self._conn
        if self._connect_failed:
            return None
        conn: sqlite3.Connection | None = None
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.commit()
            self._conn = conn
            return conn
        except Exception as exc:
            self._connect_failed = True
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            logger.debug(
                "fal job store unavailable this session (%s): %s", self._db_path, exc
            )
            return None

    # -- writes ---------------------------------------------------------------

    def record_submit(self, endpoint: str, request_id: str, note: str = "") -> None:
        """Record a freshly queued job as 'submitted'. Never raises."""
        try:
            if not request_id:
                return
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute(
                    "INSERT OR REPLACE INTO jobs "
                    "(request_id, endpoint, status, submitted_at, collected_at, note) "
                    "VALUES (?, ?, ?, ?, NULL, ?)",
                    (request_id, endpoint, _STATUS_SUBMITTED, time.time(), note),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("job store record_submit failed: %s", exc)
        self.prune(_PRUNE_AFTER_DAYS)

    def mark_collected(self, request_id: str) -> None:
        """Mark a job 'collected'. Unknown ids are inserted silently (recovery
        of jobs submitted in other sessions). Never raises."""
        try:
            if not request_id:
                return
            now = time.time()
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(request_id, endpoint, status, submitted_at, collected_at, note) "
                    "VALUES (?, '', ?, ?, ?, '')",
                    (request_id, _STATUS_COLLECTED, now, now),
                )
                conn.execute(
                    "UPDATE jobs SET status = ?, collected_at = ? WHERE request_id = ?",
                    (_STATUS_COLLECTED, now, request_id),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("job store mark_collected failed: %s", exc)

    def prune(self, older_than_days: float = _PRUNE_AFTER_DAYS) -> None:
        """Delete jobs submitted more than ``older_than_days`` ago. Never raises.

        fal queue entries expire long before this window, so stale rows are
        pure noise by then.
        """
        try:
            cutoff = time.time() - float(older_than_days) * _SECONDS_PER_DAY
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute("DELETE FROM jobs WHERE submitted_at < ?", (cutoff,))
                conn.commit()
        except Exception as exc:
            logger.debug("job store prune failed: %s", exc)

    # -- reads ----------------------------------------------------------------

    def entries(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        """Return jobs newest first as dicts keyed by column name. Never raises."""
        try:
            query = f"SELECT {', '.join(_COLUMNS)} FROM jobs"
            params: tuple[Any, ...] = ()
            if status:
                query += " WHERE status = ?"
                params = (status,)
            query += " ORDER BY submitted_at DESC LIMIT ?"
            params = (*params, int(limit))
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return []
                rows = conn.execute(query, params).fetchall()
            return [dict(zip(_COLUMNS, row)) for row in rows]
        except Exception as exc:
            logger.debug("job store entries failed: %s", exc)
            return []

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Jobs submitted but not yet collected, newest first. Never raises."""
        return self.entries(limit=limit, status=_STATUS_SUBMITTED)

    def counts(self) -> dict[str, int]:
        """Return {'submitted': n, 'collected': n}. Never raises."""
        result = {_STATUS_SUBMITTED: 0, _STATUS_COLLECTED: 0}
        try:
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return result
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                ).fetchall()
            return {**result, **{str(status): int(count) for status, count in rows if status in result}}
        except Exception as exc:
            logger.debug("job store counts failed: %s", exc)
            return result

    def report(self, limit: int = 20) -> str:
        """Multi-line human summary of the async job inbox. Never raises."""
        try:
            counts = self.counts()
            pending = counts.get(_STATUS_SUBMITTED, 0)
            collected = counts.get(_STATUS_COLLECTED, 0)
            lines = [
                f"Fal job inbox: {pending} pending, {collected} collected "
                "(async jobs survive ComfyUI restarts)"
            ]
            now = time.time()
            lines.extend(_format_entry(entry, now) for entry in self.entries(limit=limit))
            if pending == 0 and collected == 0:
                lines.append("  (empty — queue jobs with Fal Submit to fill the inbox)")
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("job store report failed: %s", exc)
            return "Fal job inbox: report unavailable"
