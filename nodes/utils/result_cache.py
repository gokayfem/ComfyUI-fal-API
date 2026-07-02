"""Persistent two-layer cache so the same fal call is never paid for twice.

Layer 1 (``results``) caches full API results keyed by endpoint + canonical
arguments. Layer 2 (``uploads``) caches fal media URLs keyed by the sha256 of
the uploaded file's bytes, so re-uploading identical content reuses the same
URL (which in turn keeps the result-cache key stable).

Both layers live in one sqlite database that survives ComfyUI restarts.
Every public method is best-effort: any sqlite/config failure degrades to a
cache miss or a no-op — bookkeeping must never break generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

from .config import FalConfig
from .logger import logger

_DB_ENV_VAR = "COMFYUI_FAL_API_CACHE_DB"
_DB_SUBDIR = "comfyui-fal-api"
_DB_FILENAME = "cache.db"

_DEFAULT_ENABLED = True
_DEFAULT_TTL_DAYS = 7  # fal CDN URLs inside cached results can expire; keep modest.
_DEFAULT_MAX_ENTRIES = 5000
_SECONDS_PER_DAY = 86400.0

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS results (
        key TEXT PRIMARY KEY,
        endpoint TEXT,
        request_id TEXT,
        result_json TEXT,
        created REAL,
        last_used REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS uploads (
        content_hash TEXT PRIMARY KEY,
        url TEXT,
        created REAL
    )
    """,
)


def _default_db_path() -> str:
    """Resolve the cache database path.

    Precedence: COMFYUI_FAL_API_CACHE_DB env var, then the ComfyUI user
    directory, then ~/.cache (when running outside ComfyUI).
    """
    override = os.environ.get(_DB_ENV_VAR)
    if override:
        return override
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, _DB_SUBDIR, _DB_FILENAME)


def _format_amount(value: float) -> str:
    """Format a dollar amount compactly (up to 4 decimals, no trailing zeros)."""
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text or "0"


class ResultCache:
    """Thread-safe singleton over the persistent result/upload cache."""

    _instance: ResultCache | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> ResultCache:
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
        self._hits = 0
        self._misses = 0

    # -- connection / config ------------------------------------------------

    def _connection(self) -> sqlite3.Connection | None:
        """Open (once) and return the sqlite connection; None if unavailable.

        Must be called with ``self._lock`` held. A corrupted or unwritable
        database disables the cache for the session instead of raising.
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
            for statement in _SCHEMA:
                conn.execute(statement)
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
                "fal cache unavailable, caching disabled this session (%s): %s",
                self._db_path,
                exc,
            )
            return None

    @staticmethod
    def _enabled() -> bool:
        try:
            return bool(FalConfig().get_setting("cache", "enabled", _DEFAULT_ENABLED))
        except Exception as exc:
            logger.debug("cache 'enabled' setting read failed: %s", exc)
            return _DEFAULT_ENABLED

    @staticmethod
    def _ttl_seconds() -> float:
        try:
            days = float(FalConfig().get_setting("cache", "ttl_days", _DEFAULT_TTL_DAYS))
        except Exception as exc:
            logger.debug("cache 'ttl_days' setting read failed: %s", exc)
            days = float(_DEFAULT_TTL_DAYS)
        return days * _SECONDS_PER_DAY

    @staticmethod
    def _max_entries() -> int:
        try:
            return int(float(FalConfig().get_setting("cache", "max_entries", _DEFAULT_MAX_ENTRIES)))
        except Exception as exc:
            logger.debug("cache 'max_entries' setting read failed: %s", exc)
            return _DEFAULT_MAX_ENTRIES

    # -- layer 1: result cache ----------------------------------------------

    @staticmethod
    def make_key(endpoint: str, arguments: dict[str, Any]) -> str:
        """Deterministic cache key: sha256 of endpoint + canonical arguments."""
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{endpoint}\x00{canonical}".encode()).hexdigest()

    def get(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Return the cached result dict for this exact call, or None on miss."""
        try:
            if not self._enabled():
                return None
            result_json = self._fetch_live_result_json(self.make_key(endpoint, arguments))
            if result_json is not None:
                result = json.loads(result_json)
                if isinstance(result, dict):
                    self._hits += 1
                    self._log_hit(endpoint)
                    return result
            self._misses += 1
            return None
        except Exception as exc:
            logger.debug("[%s] result cache get failed: %s", endpoint, exc)
            return None

    def _fetch_live_result_json(self, key: str) -> str | None:
        """Fetch a non-expired row's JSON, deleting expired rows on the way."""
        now = time.time()
        ttl_seconds = self._ttl_seconds()
        with self._lock:
            conn = self._connection()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT result_json, created FROM results WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            result_json, created = row
            if ttl_seconds > 0 and now - float(created or 0.0) > ttl_seconds:
                conn.execute("DELETE FROM results WHERE key = ?", (key,))
                conn.commit()
                return None
            conn.execute("UPDATE results SET last_used = ? WHERE key = ?", (now, key))
            conn.commit()
            return str(result_json)

    @staticmethod
    def _log_hit(endpoint: str) -> None:
        """Log a cache hit, with the estimated cost saved when pricing is known."""
        saved = ""
        try:
            from .pricing import PricingUtils

            total = PricingUtils.estimate(endpoint, 1)["total"]
            if isinstance(total, (int, float)):
                saved = f" (saved ~${_format_amount(total)})"
        except Exception:
            saved = ""
        logger.info("[%s] cache HIT%s — returning stored result, no charge", endpoint, saved)

    def put(
        self,
        endpoint: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        request_id: str | None = None,
    ) -> None:
        """Store a successful live result; prunes oldest rows beyond max_entries."""
        try:
            if not self._enabled() or not isinstance(result, dict):
                return
            key = self.make_key(endpoint, arguments)
            result_json = json.dumps(result, default=str)
            max_entries = self._max_entries()
            now = time.time()
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute(
                    "INSERT OR REPLACE INTO results "
                    "(key, endpoint, request_id, result_json, created, last_used) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, endpoint, request_id, result_json, now, now),
                )
                if max_entries > 0:
                    conn.execute(
                        "DELETE FROM results WHERE key IN "
                        "(SELECT key FROM results ORDER BY last_used DESC LIMIT -1 OFFSET ?)",
                        (max_entries,),
                    )
                conn.commit()
        except Exception as exc:
            logger.debug("[%s] result cache put failed: %s", endpoint, exc)

    def invalidate_key(self, key: str) -> None:
        """Delete one result row by its cache key. Never raises."""
        try:
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute("DELETE FROM results WHERE key = ?", (key,))
                conn.commit()
        except Exception as exc:
            logger.debug("result cache invalidate failed: %s", exc)

    def clear(self) -> None:
        """Delete all cached results and uploads. Never raises."""
        try:
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute("DELETE FROM results")
                conn.execute("DELETE FROM uploads")
                conn.commit()
        except Exception as exc:
            logger.debug("result cache clear failed: %s", exc)

    def stats(self) -> dict[str, Any]:
        """Return {entries, db_path, hits, misses}; hit/miss counts are per session."""
        entries = 0
        try:
            with self._lock:
                conn = self._connection()
                if conn is not None:
                    row = conn.execute("SELECT COUNT(*) FROM results").fetchone()
                    entries = int(row[0]) if row else 0
        except Exception as exc:
            logger.debug("result cache stats failed: %s", exc)
        return {
            "entries": entries,
            "db_path": self._db_path,
            "hits": self._hits,
            "misses": self._misses,
        }

    def find_request_by_url(self, url: str) -> dict[str, Any] | None:
        """Find the origin of a result URL: {"endpoint_id", "request_id"} or None.

        Scans cached results (most recently used first) for one whose JSON
        contains ``url`` as an exact substring. Only rows that recorded a
        request_id qualify. Best-effort: any failure is a miss.
        """
        try:
            target = (url or "").strip()
            if not target:
                return None
            # LIKE treats %, _ (and our escape char) specially — escape them
            # so URLs containing percent-encoding still match literally.
            escaped = (
                target.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return None
                row = conn.execute(
                    "SELECT endpoint, request_id FROM results "
                    "WHERE request_id IS NOT NULL "
                    "AND result_json LIKE '%' || ? || '%' ESCAPE '\\' "
                    "ORDER BY last_used DESC LIMIT 1",
                    (escaped,),
                ).fetchone()
            if row is None:
                return None
            endpoint, request_id = row
            return {"endpoint_id": endpoint, "request_id": request_id}
        except Exception as exc:
            logger.debug("result cache url lookup failed: %s", exc)
            return None

    # -- layer 2: upload cache ----------------------------------------------

    def get_upload(self, content_hash: str) -> str | None:
        """Return the cached fal URL for previously uploaded content, or None."""
        try:
            if not self._enabled():
                return None
            now = time.time()
            ttl_seconds = self._ttl_seconds()
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return None
                row = conn.execute(
                    "SELECT url, created FROM uploads WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
                if row is None:
                    return None
                url, created = row
                if ttl_seconds > 0 and now - float(created or 0.0) > ttl_seconds:
                    conn.execute(
                        "DELETE FROM uploads WHERE content_hash = ?", (content_hash,)
                    )
                    conn.commit()
                    return None
                return str(url) if url else None
        except Exception as exc:
            logger.debug("upload cache get failed: %s", exc)
            return None

    def put_upload(self, content_hash: str, url: str) -> None:
        """Remember that this content hash uploaded to ``url``. Never raises."""
        try:
            if not self._enabled():
                return
            with self._lock:
                conn = self._connection()
                if conn is None:
                    return
                conn.execute(
                    "INSERT OR REPLACE INTO uploads (content_hash, url, created) "
                    "VALUES (?, ?, ?)",
                    (content_hash, url, time.time()),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("upload cache put failed: %s", exc)
