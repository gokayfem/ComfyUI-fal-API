"""fal.ai platform billing: account balance, usage reconciliation, spend guard.

Uses the fal Platform APIs (base ``https://api.fal.ai/v1``):

- ``GET /account/billing?expand=credits`` — current credit balance.
- ``GET /models/requests/by-endpoint`` — per-request records (request_id,
  endpoint_id; fal does not publish a per-request billed amount).
- ``GET /models/usage?expand=summary`` — aggregated billed cost per endpoint.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .config import FalConfig
from .errors import FalApiError
from .ledger import SessionLedger
from .logger import logger

_API_BASE = "https://api.fal.ai/v1"
_REQUEST_TIMEOUT = (5, 15)
_BALANCE_CACHE_TTL_S = 60.0
_MAX_ENDPOINT_FILTERS = 50
_MAX_REQUEST_LIMIT = 100
_BILLING_DASHBOARD_URL = "https://fal.ai/dashboard/billing"

_balance_lock = threading.Lock()
# [cached balance (float|None), fetched_at unix time]; fetched_at 0 = no fetch yet.
_balance_cache: list[Any] = [None, 0.0]

_warn_lock = threading.Lock()
_warned_once: frozenset[str] = frozenset()


def _warn_once(topic: str, message: str) -> None:
    """Log ``message`` as WARNING the first time per topic, DEBUG afterwards."""
    global _warned_once
    with _warn_lock:
        first_time = topic not in _warned_once
        _warned_once = _warned_once | {topic}
    if first_time:
        logger.warning(message)
    else:
        logger.debug(message)


def _get_json(path: str, params: dict[str, Any], topic: str) -> Any | None:
    """GET a Platform API path; return parsed JSON or None on any failure."""
    key = FalConfig().get_key()
    if not key:
        _warn_once(topic, f"Cannot call fal Platform API {path}: FAL_KEY is not configured")
        return None
    try:
        response = requests.get(
            f"{_API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Key {key}"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        _warn_once(topic, f"fal Platform API {path} unavailable: {exc}")
        return None


def _fetch_balance() -> float | None:
    """Fetch the current credit balance in USD, or None on any failure."""
    payload = _get_json("/account/billing", {"expand": "credits"}, topic="balance")
    if not isinstance(payload, dict):
        return None
    credits = payload.get("credits")
    balance = credits.get("current_balance") if isinstance(credits, dict) else None
    if isinstance(balance, (int, float)):
        return float(balance)
    _warn_once("balance", "fal balance response had no credits.current_balance field")
    return None


def _ledger_endpoints(entries: list[dict[str, Any]]) -> list[str]:
    """Unique endpoint ids from ledger entries, capped at the API filter limit."""
    seen: list[str] = []
    for entry in entries:
        endpoint = entry.get("endpoint_id")
        if isinstance(endpoint, str) and endpoint and endpoint not in seen:
            seen.append(endpoint)
    return seen[:_MAX_ENDPOINT_FILTERS]


def _earliest_timestamp_iso(entries: list[dict[str, Any]]) -> str | None:
    """ISO8601 UTC timestamp of the earliest ledger entry, or None."""
    stamps = [e["timestamp"] for e in entries if isinstance(e.get("timestamp"), (int, float))]
    if not stamps:
        return None
    earliest = datetime.fromtimestamp(min(stamps), tz=timezone.utc)
    return earliest.strftime("%Y-%m-%dT%H:%M:%SZ")


def _billed_total_since(entries: list[dict[str, Any]]) -> float | None:
    """Aggregated billed cost for the ledger's endpoints since the session start.

    Best effort via ``GET /models/usage?expand=summary``; the window covers the
    whole workspace, so concurrent non-session calls may be included.
    """
    endpoints = _ledger_endpoints(entries)
    start = _earliest_timestamp_iso(entries)
    if not endpoints or start is None:
        return None
    params = {
        "expand": "summary",
        "start": start,
        "endpoint_id": ",".join(endpoints),
        "bound_to_timeframe": "false",
    }
    payload = _get_json("/models/usage", params, topic="usage")
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), list):
        return None
    costs = [
        item.get("cost")
        for item in payload["summary"]
        if isinstance(item, dict) and isinstance(item.get("cost"), (int, float))
    ]
    return float(sum(costs)) if costs else None


class BillingUtils:
    """Read-only access to fal account balance and billed usage. Never raises."""

    @staticmethod
    def get_balance(force: bool = False) -> float | None:
        """Current account credit balance in USD, or None on any failure.

        Results (including failures) are cached for 60 seconds so callers such
        as SpendGuard.preflight do not hammer the API; ``force=True`` bypasses.
        """
        global _balance_cache
        now = time.time()
        if not force:
            with _balance_lock:
                value, fetched_at = _balance_cache
            if fetched_at > 0 and now - fetched_at < _BALANCE_CACHE_TTL_S:
                return value
        value = _fetch_balance()
        with _balance_lock:
            _balance_cache = [value, time.time()]
        return value

    @staticmethod
    def get_recent_usage(limit: int = 50) -> list[dict[str, Any]] | None:
        """Recent per-request records for this session's endpoints, or None.

        Backed by ``GET /models/requests/by-endpoint`` filtered to the
        endpoints recorded in the SessionLedger. fal's Platform APIs do not
        expose a per-request billed amount (usage is aggregated), so ``amount``
        is always None. Returns None when the ledger is empty or the API is
        unavailable.
        """
        try:
            endpoints = _ledger_endpoints(SessionLedger().entries())
            if not endpoints:
                return None
            params = {
                "endpoint_id": ",".join(endpoints),
                "limit": max(1, min(int(limit), _MAX_REQUEST_LIMIT)),
            }
            payload = _get_json("/models/requests/by-endpoint", params, topic="requests")
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                return None
            return [
                {
                    "request_id": item.get("request_id"),
                    "endpoint": item.get("endpoint_id"),
                    "amount": None,
                }
                for item in payload["items"]
                if isinstance(item, dict)
            ]
        except Exception as exc:
            logger.debug("get_recent_usage failed: %s", exc)
            return None

    @staticmethod
    def reconcile_ledger() -> dict[str, Any]:
        """Best-effort reconciliation of the SessionLedger against fal's records.

        Returns ``{"matched": n, "billed_total": float|None, "estimated_total": float}``
        where ``matched`` counts ledger request_ids confirmed by the requests
        API and ``billed_total`` is the aggregated billed cost (None when the
        usage API is unavailable). Never raises.
        """
        estimated_total = SessionLedger().total_cost()
        result: dict[str, Any] = {
            "matched": 0,
            "billed_total": None,
            "estimated_total": estimated_total,
        }
        try:
            entries = SessionLedger().entries()
            if not entries:
                return result
            ledger_ids = {e["request_id"] for e in entries if e.get("request_id")}
            usage = BillingUtils.get_recent_usage(limit=_MAX_REQUEST_LIMIT) or []
            matched = sum(1 for record in usage if record.get("request_id") in ledger_ids)
            return {
                **result,
                "matched": matched,
                "billed_total": _billed_total_since(entries),
            }
        except Exception as exc:
            logger.debug("reconcile_ledger failed: %s", exc)
            return result


def _setting_float(name: str) -> float:
    """Read a [spend_guard] float setting; 0.0 when unset or unparseable."""
    try:
        value = FalConfig().get_setting("spend_guard", name, 0)
        if isinstance(value, bool):
            return 0.0
        return float(value)
    except Exception as exc:
        logger.debug("Invalid [spend_guard] %s value: %s", name, exc)
        return 0.0


class SpendGuard:
    """Pre-call spend checks configured via config.ini [spend_guard]."""

    @staticmethod
    def settings() -> dict[str, float]:
        """Active spend-guard settings (0.0 means the check is disabled)."""
        return {
            "session_budget_usd": _setting_float("session_budget_usd"),
            "min_balance_usd": _setting_float("min_balance_usd"),
        }

    @staticmethod
    def preflight(endpoint: str) -> None:
        """Raise FalApiError if a configured spend limit blocks this call.

        Frozen contract: called by ApiHandler before every fal request. Checks
        are skipped when their setting is 0/unset; an unavailable balance API
        never blocks. Raises nothing except FalApiError.
        """
        budget = _setting_float("session_budget_usd")
        if budget > 0:
            spent = SessionLedger().total_cost()
            if spent >= budget:
                raise FalApiError(
                    "spend-guard",
                    f"Session budget ${budget:.2f} reached (spent ~${spent:.2f}). "
                    "Raise [spend_guard] session_budget_usd in config.ini or "
                    "reset the session ledger.",
                )

        floor = _setting_float("min_balance_usd")
        if floor > 0:
            balance = BillingUtils.get_balance()
            if balance is None:
                logger.debug(
                    "Spend guard: balance unavailable; allowing request to %s", endpoint
                )
            elif balance < floor:
                raise FalApiError(
                    "spend-guard",
                    f"fal balance ${balance:.2f} is below your ${floor:.2f} floor "
                    f"— top up at {_BILLING_DASHBOARD_URL}",
                )
