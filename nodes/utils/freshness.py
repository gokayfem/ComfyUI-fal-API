"""Checks the live fal.ai catalog for models missing from the local registry.

``check_for_new_models`` diffs the public catalog against the committed
``data/fal_registry.json`` and caches the result module-level (1h TTL) so the
sidebar and the startup check share one fetch. ``schedule_startup_check``
spawns a delayed daemon thread that logs a single INFO line when the local
registry is behind. Nothing in here may break node loading: the startup path
never raises.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from .logger import logger

CATALOG_URL = "https://fal.ai/api/models?page={page}&total={total}"
_USER_AGENT = "ComfyUI-fal-API-freshness/1.0"
_PAGE_SIZE = 100
_MAX_PAGES = 25
_MAX_NEW_LISTED = 25
_CACHE_TTL_S = 3600.0
_STARTUP_DELAY_S = 10.0
_DEFAULT_TIMEOUT_S = 20.0

_lock = threading.Lock()
_cached_result: dict[str, Any] | None = None
_startup_scheduled = False


def _registry_path() -> str:
    """Path to data/fal_registry.json at the repo root."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(utils_dir))
    return os.path.join(repo_root, "data", "fal_registry.json")


def _registry_endpoint_ids() -> set[str]:
    """Endpoint ids present in the committed registry; empty set on failure."""
    try:
        with open(_registry_path(), encoding="utf-8") as handle:
            registry = json.load(handle)
        models = registry.get("models")
        if not isinstance(models, list):
            raise ValueError("'models' is not a list")
        return {
            str(model["endpoint_id"])
            for model in models
            if isinstance(model, dict) and model.get("endpoint_id")
        }
    except Exception as err:
        logger.debug("freshness: could not read local registry: %s", err)
        return set()


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    """Normalize one catalog API page into a list of item dicts."""
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = next(
            (
                payload[key]
                for key in ("items", "models", "data", "results")
                if isinstance(payload.get(key), list)
            ),
            [],
        )
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def _fetch_catalog(timeout_s: float) -> list[dict[str, Any]]:
    """Fetch catalog pages until an empty page (hard cap _MAX_PAGES).

    Raises RuntimeError when the very first page cannot be fetched; a failure
    on a later page returns the partial catalog (better a lower bound than
    nothing).
    """
    import requests

    items: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        url = CATALOG_URL.format(page=page, total=_PAGE_SIZE)
        try:
            response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout_s)
            response.raise_for_status()
            page_items = _extract_items(response.json())
        except Exception as err:
            if page == 1:
                raise RuntimeError(f"fal catalog fetch failed: {err}") from err
            logger.debug("freshness: catalog page %d failed (%s); using partial catalog", page, err)
            break
        if not page_items:
            break
        items = items + page_items
    return items


def _is_live_public(item: dict[str, Any]) -> bool:
    return bool(
        item.get("id")
        and item.get("status") == "public"
        and not item.get("deprecated")
        and not item.get("removed")
    )


def _new_model_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint_id": str(item.get("id") or ""),
        "title": str(item.get("title") or "").strip(),
        "category": str(item.get("category") or "").strip(),
        "published_at": str(item.get("publishedAt") or item.get("date") or "").strip(),
    }


def check_for_new_models(timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Diff the live fal catalog against the local registry (cached, 1h TTL).

    Returns ``{"new_count", "new_models" (newest first, max 25), "checked_at"}``.
    Raises RuntimeError when the catalog cannot be reached at all; failed runs
    are never cached.
    """
    global _cached_result
    with _lock:
        if (
            _cached_result is not None
            and time.time() - float(_cached_result.get("checked_at", 0)) < _CACHE_TTL_S
        ):
            return _cached_result

    known_ids = _registry_endpoint_ids()
    catalog = _fetch_catalog(timeout_s)
    live = [item for item in catalog if _is_live_public(item)]

    seen: set[str] = set()
    fresh: list[dict[str, Any]] = []
    for item in live:
        endpoint_id = str(item["id"])
        if endpoint_id in known_ids or endpoint_id in seen:
            continue
        seen.add(endpoint_id)
        fresh = fresh + [_new_model_entry(item)]

    fresh.sort(key=lambda entry: entry["published_at"], reverse=True)
    result = {
        "new_count": len(fresh),
        "new_models": fresh[:_MAX_NEW_LISTED],
        "checked_at": time.time(),
    }

    with _lock:
        _cached_result = result
    return result


def _startup_check_enabled() -> bool:
    if os.environ.get("FAL_DISABLE_STARTUP_CHECK"):
        return False
    try:
        from .config import FalConfig

        value = FalConfig().get_setting("registry", "startup_check", True)
    except Exception as err:
        logger.debug("freshness: could not read startup_check setting: %s", err)
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _startup_worker() -> None:
    """Delayed freshness check; logs one INFO line, never raises."""
    try:
        time.sleep(_STARTUP_DELAY_S)
        result = check_for_new_models()
        new_count = result.get("new_count", 0)
        if new_count:
            logger.info(
                "fal catalog: %d models newer than the local registry — "
                "see the fal sidebar or run scripts/build_registry.py",
                new_count,
            )
        else:
            logger.debug("fal catalog: local registry is up to date")
    except Exception as err:
        logger.debug("fal registry freshness check failed: %s", err)


def schedule_startup_check() -> bool:
    """Spawn the delayed startup freshness thread once. Never raises.

    Returns True when a thread was started (enabled and not yet scheduled).
    """
    global _startup_scheduled
    try:
        with _lock:
            if _startup_scheduled:
                return False
            _startup_scheduled = True
        if not _startup_check_enabled():
            logger.debug("freshness: startup check disabled via config")
            return False
        thread = threading.Thread(
            target=_startup_worker, name="fal-registry-freshness", daemon=True
        )
        thread.start()
        return True
    except Exception as err:
        logger.debug("freshness: could not schedule startup check: %s", err)
        return False
