"""HTTP routes exposing fal pricing, session costs, jobs and balance to the ComfyUI frontend.

Registered on ComfyUI's PromptServer under ``/fal_api/*``. The module must
import cleanly without ComfyUI (headless/tests): ``register()`` is a no-op when
``server`` is unavailable, and every handler is a thin wrapper over a pure
function so the payload logic is unit-testable without aiohttp.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from .utils.billing import BillingUtils
from .utils.ledger import SessionLedger
from .utils.logger import logger
from .utils.pricing import PricingUtils

_SESSION_TAIL = 20
_DEFAULT_JOB_LIMIT = 50
_DEFAULT_SEARCH_LIMIT = 25
_MAX_SEARCH_LIMIT = 100

_cache_lock = threading.Lock()
# [(model, pricing_info|None), ...] in registry order; None until first build.
_catalog_cache: list[tuple[dict[str, Any], dict[str, Any] | None]] | None = None
_pricing_map_cache: dict[str, dict[str, Any]] | None = None


# -- registry catalog (lazy, built once) ---------------------------------------


def _registry_path() -> str:
    """Path to data/fal_registry.json at the repo root."""
    nodes_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(nodes_dir), "data", "fal_registry.json")


def _read_models() -> list[dict[str, Any]]:
    """Read registry models; empty list on any failure."""
    try:
        with open(_registry_path(), encoding="utf-8") as handle:
            registry = json.load(handle)
        models = registry.get("models")
        if not isinstance(models, list):
            raise ValueError("'models' is not a list")
        return [m for m in models if isinstance(m, dict) and m.get("endpoint_id")]
    except Exception as exc:
        logger.warning("server_routes: could not load fal registry: %s", exc)
        return []


def _format_amount(value: float) -> str:
    """Format a dollar amount compactly (up to 4 decimals, no trailing zeros)."""
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _pricing_info(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a PricingUtils.parse result into {"label", "per_run"}, or None.

    per_run known -> "≈$X/run"; only per_unit -> "$X per <unit>"; else None.
    """
    per_run = parsed.get("per_run")
    if isinstance(per_run, (int, float)):
        return {"label": f"≈${_format_amount(float(per_run))}/run", "per_run": float(per_run)}
    per_unit = parsed.get("per_unit")
    unit = parsed.get("unit")
    if isinstance(per_unit, (int, float)) and unit:
        return {"label": f"${_format_amount(float(per_unit))} per {unit}", "per_run": None}
    return None


def _node_key_for(model: dict[str, Any]) -> str:
    """Dynamic node class key for a registry model (same as factory.node_key)."""
    try:
        from .dynamic.factory import node_key

        return node_key(model)
    except Exception:  # stripped env without the dynamic package's deps
        return "FalAPI_" + str(model.get("endpoint_id", "")).replace("/", "-")


def _catalog() -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Registry models paired with parsed pricing info, cached after first build."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    with _cache_lock:
        if _catalog_cache is not None:
            return _catalog_cache
        _catalog_cache = [
            (model, _pricing_info(PricingUtils.parse(str(model.get("pricing") or ""))))
            for model in _read_models()
        ]
        return _catalog_cache


# -- pure payload builders (unit-tested directly) -------------------------------


def _pricing_map() -> dict[str, dict[str, Any]]:
    """{node_class_key: {"label", "per_run"}} for every priced dynamic node."""
    global _pricing_map_cache
    if _pricing_map_cache is not None:
        return _pricing_map_cache
    mapping = {
        _node_key_for(model): info for model, info in _catalog() if info is not None
    }
    with _cache_lock:
        _pricing_map_cache = mapping
    return mapping


def _pricing_single(endpoint_id: str) -> dict[str, Any]:
    """Live pricing label for one endpoint; {"label": None} when unknown."""
    endpoint = (endpoint_id or "").strip()
    if not endpoint:
        return {"label": None, "per_run": None}
    estimate = PricingUtils.estimate(endpoint)
    per_run = estimate.get("per_run")
    if isinstance(per_run, (int, float)):
        return {"label": f"≈${_format_amount(float(per_run))}/run", "per_run": float(per_run)}
    unit_note = estimate.get("unit_note") or ""
    if unit_note:
        return {"label": unit_note, "per_run": None}
    return {"label": None, "per_run": None}


def _session() -> dict[str, Any]:
    """Session ledger totals plus the last few call entries."""
    ledger = SessionLedger()
    entries = ledger.entries()
    return {
        "total_usd": ledger.total_cost(),
        "calls": len(entries),
        "entries": entries[-_SESSION_TAIL:],
    }


def _jobs(limit: int = _DEFAULT_JOB_LIMIT) -> dict[str, Any]:
    """Persistent async-job inbox; degrades to empty when the store is missing."""
    try:
        from .utils.job_store import JobStore

        store = JobStore()
        return {"jobs": store.entries(limit=limit), "counts": store.counts()}
    except Exception as exc:
        logger.debug("server_routes: job store unavailable: %s", exc)
        return {"jobs": [], "counts": {}}


def _balance() -> dict[str, Any]:
    """Account credit balance (60s-cached inside BillingUtils)."""
    return {"balance_usd": BillingUtils.get_balance()}


def _search_models(
    q: str = "",
    category: str = "",
    max_price: float | None = None,
    limit: int = _DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """Search the registry; newest first, filtered by text/category/per-run price."""
    needle = (q or "").strip().lower()
    wanted_category = (category or "").strip()
    capped = max(1, min(int(limit), _MAX_SEARCH_LIMIT))

    def matches(model: dict[str, Any], info: dict[str, Any] | None) -> bool:
        haystack = f"{model.get('endpoint_id', '')} {model.get('title', '')}".lower()
        if needle and needle not in haystack:
            return False
        if wanted_category and model.get("category") != wanted_category:
            return False
        if max_price is not None:
            per_run = (info or {}).get("per_run")
            if not isinstance(per_run, (int, float)) or per_run > max_price:
                return False
        return True

    hits = [(model, info) for model, info in _catalog() if matches(model, info)]
    hits.sort(key=lambda pair: str(pair[0].get("published_at") or ""), reverse=True)
    return [
        {
            "endpoint_id": model["endpoint_id"],
            "title": model.get("title") or model["endpoint_id"],
            "category": model.get("category"),
            "label": (info or {}).get("label"),
            "thumbnail": model.get("thumbnail") or None,
        }
        for model, info in hits[:capped]
    ]


# -- registry freshness + refresh -----------------------------------------------

_RESTART_NOTE = "Restart ComfyUI after the refresh finishes: new nodes register at import time."
_REFRESH_TIMEOUT_S = 1800

_refresh_lock = threading.Lock()
_refresh_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "ok": None,
    "message": "Registry refresh has not been started.",
}


def _repo_root() -> str:
    nodes_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(nodes_dir)


def _registry_status() -> dict[str, Any]:
    """Cached diff of the live fal catalog vs. the local registry (may fetch)."""
    from .utils.freshness import check_for_new_models

    return check_for_new_models(timeout_s=20)


def _refresh_status() -> dict[str, Any]:
    """Snapshot of the background registry-refresh state."""
    with _refresh_lock:
        return {**_refresh_state, "restart_note": _RESTART_NOTE}


def _run_refresh_subprocess() -> tuple[bool, str]:
    """Run scripts/build_registry.py; returns (ok, message)."""
    import subprocess
    import sys

    root = _repo_root()
    command = [
        sys.executable,
        os.path.join(root, "scripts", "build_registry.py"),
        "--out",
        os.path.join("data", "fal_registry.json"),
    ]
    completed = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=_REFRESH_TIMEOUT_S
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-500:]
        return False, f"build_registry.py exited with {completed.returncode}: {tail}"
    return True, f"Registry refreshed. {_RESTART_NOTE}"


def _finish_refresh(ok: bool, message: str) -> None:
    global _refresh_state
    with _refresh_lock:
        _refresh_state = {
            **_refresh_state,
            "running": False,
            "finished_at": time.time(),
            "ok": ok,
            "message": message,
        }


def _refresh_worker(runner: Callable[[], tuple[bool, str]]) -> None:
    """Run the refresh and record the outcome. Never raises."""
    try:
        ok, message = runner()
    except Exception as exc:
        logger.warning("server_routes: registry refresh failed: %s", exc)
        ok, message = False, f"Registry refresh failed: {exc}"
    _finish_refresh(ok, message)
    logger.info("server_routes: registry refresh finished (ok=%s): %s", ok, message)


def _start_refresh(
    runner: Callable[[], tuple[bool, str]] | None = None,
    spawn: Callable[[Callable[[], None]], None] | None = None,
) -> dict[str, Any]:
    """Start a background registry rebuild; no-op when one is already running.

    ``runner``/``spawn`` are injectable for tests (stub subprocess / run inline).
    """
    global _refresh_state
    with _refresh_lock:
        if _refresh_state["running"]:
            return {"started": False, **_refresh_state, "restart_note": _RESTART_NOTE}
        _refresh_state = {
            **_refresh_state,
            "running": True,
            "started_at": time.time(),
            "finished_at": None,
            "ok": None,
            "message": "Registry refresh running — rebuilding data/fal_registry.json...",
        }

    active_runner = runner or _run_refresh_subprocess

    def work() -> None:
        _refresh_worker(active_runner)

    if spawn is not None:
        spawn(work)
    else:
        threading.Thread(target=work, name="fal-registry-refresh", daemon=True).start()
    return {"started": True, **_refresh_status()}


def _cancel(endpoint_id: str, request_id: str) -> dict[str, Any]:
    """Best-effort cancel of a queued fal request via fal_client. Never raises."""
    endpoint = (endpoint_id or "").strip()
    request = (request_id or "").strip()
    if not endpoint or not request:
        return {"ok": False, "error": "endpoint_id and request_id are required"}
    try:
        from .utils.config import FalConfig

        FalConfig().get_client().cancel(endpoint, request)
        logger.info("server_routes: cancelled %s request %s", endpoint, request)
        return {"ok": True}
    except Exception as exc:
        logger.debug("server_routes: cancel %s/%s failed: %s", endpoint, request, exc)
        return {"ok": False, "error": str(exc)}


# -- aiohttp glue ----------------------------------------------------------------


def _json_response(payload: Any, status: int = 200) -> Any:
    """aiohttp JSON response; the raw payload when aiohttp is unavailable (tests)."""
    try:
        from aiohttp import web
    except ImportError:
        return payload
    return web.json_response(payload, status=status)


def _guarded(build: Callable[[], Any], route: str) -> Any:
    """Run a payload builder; any exception becomes a 500 {"error": ...} JSON."""
    try:
        return _json_response(build())
    except Exception as exc:
        logger.warning("server_routes: %s failed: %s", route, exc)
        return _json_response({"error": str(exc)}, status=500)


def _query_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _query_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def pricing_map_route(request: Any) -> Any:
    return _guarded(_pricing_map, "/fal_api/pricing_map")


async def pricing_route(request: Any) -> Any:
    endpoint_id = request.query.get("endpoint_id", "")
    return _guarded(lambda: _pricing_single(endpoint_id), "/fal_api/pricing")


async def session_route(request: Any) -> Any:
    return _guarded(_session, "/fal_api/session")


async def jobs_route(request: Any) -> Any:
    limit = _query_int(request.query.get("limit"), _DEFAULT_JOB_LIMIT)
    limit = max(1, min(limit, _MAX_SEARCH_LIMIT))
    return _guarded(lambda: _jobs(limit=limit), "/fal_api/jobs")


async def balance_route(request: Any) -> Any:
    return _guarded(_balance, "/fal_api/balance")


async def models_route(request: Any) -> Any:
    query = request.query
    q = query.get("q", "")
    category = query.get("category", "")
    max_price = _query_float(query.get("max_price"))
    limit = _query_int(query.get("limit"), _DEFAULT_SEARCH_LIMIT)
    return _guarded(
        lambda: _search_models(q=q, category=category, max_price=max_price, limit=limit),
        "/fal_api/models",
    )


async def registry_status_route(request: Any) -> Any:
    return _guarded(_registry_status, "/fal_api/registry_status")


async def registry_refresh_start_route(request: Any) -> Any:
    return _guarded(_start_refresh, "/fal_api/registry_refresh")


async def registry_refresh_status_route(request: Any) -> Any:
    return _guarded(_refresh_status, "/fal_api/registry_refresh")


async def cancel_route(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        body = {}
    payload = body if isinstance(body, dict) else {}
    return _guarded(
        lambda: _cancel(payload.get("endpoint_id", ""), payload.get("request_id", "")),
        "/fal_api/cancel",
    )


ROUTES: tuple[tuple[str, str, Callable[..., Any]], ...] = (
    ("GET", "/fal_api/pricing_map", pricing_map_route),
    ("GET", "/fal_api/pricing", pricing_route),
    ("GET", "/fal_api/session", session_route),
    ("GET", "/fal_api/jobs", jobs_route),
    ("GET", "/fal_api/balance", balance_route),
    ("GET", "/fal_api/models", models_route),
    ("GET", "/fal_api/registry_status", registry_status_route),
    ("GET", "/fal_api/registry_refresh", registry_refresh_status_route),
    ("POST", "/fal_api/registry_refresh", registry_refresh_start_route),
    ("POST", "/fal_api/cancel", cancel_route),
)


def register() -> bool:
    """Attach the /fal_api routes to ComfyUI's PromptServer. Never raises.

    Returns False (with a debug log) when running headless without ComfyUI.
    """
    try:
        from server import PromptServer
    except ImportError:
        logger.debug("server_routes: ComfyUI server not available; routes not registered")
        return False
    try:
        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            logger.debug("server_routes: PromptServer has no instance yet; skipping")
            return False
        routes = instance.routes
        for method, path, handler in ROUTES:
            adder = routes.get if method == "GET" else routes.post
            adder(path)(handler)
        logger.info("server_routes: registered %d /fal_api routes", len(ROUTES))
        return True
    except Exception as exc:
        logger.warning("server_routes: could not register /fal_api routes: %s", exc)
        return False


register()
