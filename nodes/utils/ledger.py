"""Thread-safe in-memory ledger of fal API calls made this ComfyUI session."""

from __future__ import annotations

import threading
import time
from typing import Any

from .logger import logger

_REPORT_TAIL = 20


def _format_cost(est_cost: Any) -> str:
    """Render an estimated cost, or a placeholder when pricing is unknown."""
    if isinstance(est_cost, (int, float)):
        return f"~${est_cost:,.4f}".rstrip("0").rstrip(".")
    return "cost unknown"


def _format_entry(entry: dict[str, Any]) -> str:
    """One report line: '  #12 fal-ai/kling.../v3 12.4s ~$0.35 req=abc123'."""
    duration = entry.get("duration_s")
    duration_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "?s"
    request_id = entry.get("request_id") or "-"
    return (
        f"  #{entry.get('index', '?')} {entry.get('endpoint_id', 'unknown')} "
        f"{duration_text} {_format_cost(entry.get('est_cost'))} req={request_id}"
    )


class SessionLedger:
    """Singleton recording every fal call this session. Methods never raise."""

    _instance: SessionLedger | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> SessionLedger:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._next_index = 1

    def record(
        self,
        endpoint_id: str,
        request_id: str | None,
        duration_s: float,
        est_cost: float | None,
    ) -> None:
        """Append one call record (indexed, timestamped). Never raises."""
        try:
            with self._lock:
                entry = {
                    "index": self._next_index,
                    "timestamp": time.time(),
                    "endpoint_id": endpoint_id,
                    "request_id": request_id,
                    "duration_s": duration_s,
                    "est_cost": est_cost,
                }
                self._entries = [*self._entries, entry]
                self._next_index += 1
        except Exception as exc:
            logger.debug("SessionLedger.record failed: %s", exc)

    def entries(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded entries."""
        try:
            with self._lock:
                return [dict(entry) for entry in self._entries]
        except Exception as exc:
            logger.debug("SessionLedger.entries failed: %s", exc)
            return []

    def total_cost(self) -> float:
        """Sum of all known estimated costs (unknowns excluded)."""
        try:
            return sum(
                entry["est_cost"]
                for entry in self.entries()
                if isinstance(entry.get("est_cost"), (int, float))
            )
        except Exception as exc:
            logger.debug("SessionLedger.total_cost failed: %s", exc)
            return 0.0

    def unknown_cost_count(self) -> int:
        """Number of recorded calls with no pricing estimate."""
        try:
            return sum(1 for entry in self.entries() if entry.get("est_cost") is None)
        except Exception as exc:
            logger.debug("SessionLedger.unknown_cost_count failed: %s", exc)
            return 0

    def reset(self) -> None:
        """Clear all entries and restart indexing."""
        try:
            with self._lock:
                self._entries = []
                self._next_index = 1
        except Exception as exc:
            logger.debug("SessionLedger.reset failed: %s", exc)

    def report(self) -> str:
        """Multi-line human summary of session usage. Never raises."""
        try:
            entries = self.entries()
            lines = [
                f"Session fal usage: {len(entries)} calls, "
                f"~${self.total_cost():,.2f} estimated, "
                f"{self.unknown_cost_count()} with unknown pricing"
            ]
            lines.extend(_format_entry(entry) for entry in entries[-_REPORT_TAIL:])
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("SessionLedger.report failed: %s", exc)
            return "Session fal usage: report unavailable"
