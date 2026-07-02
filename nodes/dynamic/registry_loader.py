"""Loads the fal model registry and builds dynamic node mappings.

Must never raise: any failure results in empty (or partial) mappings so the
static nodes keep loading no matter what.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..fal_utils import FalConfig, logger
from .any_endpoint import ANY_ENDPOINT_DISPLAY_NAME, ANY_ENDPOINT_KEY, FalAnyEndpoint
from .factory import build_display_name, build_node_class, node_key

_REGISTRY_FILENAME = "fal_registry.json"
_FIXTURE_FILENAME = "_fixture_registry.json"
_FEATURED_FILENAME = "featured_models.json"

Mappings = tuple[dict[str, type], dict[str, str]]


def _registry_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    real = package_dir.parents[1] / "data" / _REGISTRY_FILENAME
    if real.is_file():
        return real
    return package_dir / _FIXTURE_FILENAME


def _featured_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    return package_dir.parents[1] / "data" / _FEATURED_FILENAME


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _get_setting(section: str, name: str, default: Any) -> Any:
    config = FalConfig()
    getter = getattr(config, "get_setting", None)
    if getter is None:
        return default
    return getter(section, name, default)


def _category_filter() -> set[str]:
    raw = _get_setting("dynamic_nodes", "categories", "") or ""
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _read_models() -> list[dict[str, Any]]:
    path = _registry_path()
    try:
        with open(path, encoding="utf-8") as handle:
            registry = json.load(handle)
        models = registry.get("models", [])
        if not isinstance(models, list):
            raise ValueError("'models' is not a list")
        return models
    except Exception as err:
        logger.error("Failed to read fal registry at %s: %s", path, err)
        return []


def _read_featured() -> dict[str, str | None]:
    """Curated featured tier: {endpoint_id: display_name_override_or_None}.

    Empty dict when the tier is disabled, the file is missing, or unreadable.
    """
    if not _truthy(_get_setting("dynamic_nodes", "featured_tier", True)):
        logger.info("Featured fal node tier disabled via config")
        return {}
    path = _featured_path()
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        entries = document.get("featured", [])
        if not isinstance(entries, list):
            raise ValueError("'featured' is not a list")
        featured: dict[str, str | None] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("endpoint_id"):
                continue
            override = entry.get("display_name")
            featured = {
                **featured,
                str(entry["endpoint_id"]): str(override) if override else None,
            }
        return featured
    except Exception as err:
        logger.debug("No featured fal models applied (%s): %s", path, err)
        return {}


def _superseded_map(models: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """{endpoint_id: (newest_endpoint_id, newest_published_date)} per family.

    Conservative: models are grouped by (family, category) only when the
    registry declares a non-empty ``family`` (no fuzzy title matching), and a
    model is flagged only when its group has >1 member and its published_at is
    strictly older than the group's newest.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in models:
        family = str(model.get("family") or "").strip()
        if not family or not model.get("endpoint_id"):
            continue
        group_key = (family, str(model.get("category") or ""))
        groups = {**groups, group_key: groups.get(group_key, []) + [model]}

    superseded: dict[str, tuple[str, str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        newest = max(members, key=lambda m: str(m.get("published_at") or ""))
        newest_date = str(newest.get("published_at") or "")
        if not newest_date:
            continue
        for model in members:
            if str(model.get("published_at") or "") < newest_date:
                superseded = {
                    **superseded,
                    str(model["endpoint_id"]): (str(newest["endpoint_id"]), newest_date[:10]),
                }
    return superseded


def _apply_superseded_note(node_class: type, newest_id: str, newest_date: str) -> None:
    """Prefix the class DESCRIPTION with a newer-release warning."""
    note = f"Superseded: a newer release exists in this family: {newest_id} ({newest_date})"
    existing = str(getattr(node_class, "DESCRIPTION", "") or "")
    node_class.DESCRIPTION = f"{note}\n\n{existing}".rstrip()


def _unique_display_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    counter = 2
    while f"{name} #{counter}" in used:
        counter += 1
    return f"{name} #{counter}"


def _build_model_mappings(
    models: list[dict[str, Any]],
    categories: set[str],
    featured: dict[str, str | None] | None = None,
    superseded: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict[str, type], dict[str, str], int, int]:
    classes: dict[str, type] = {}
    display: dict[str, str] = {}
    used_names: set[str] = {ANY_ENDPOINT_DISPLAY_NAME}
    featured = featured or {}
    superseded = superseded or {}
    skipped = 0
    flagged = 0

    for model in models:
        try:
            if categories and model.get("category") not in categories:
                continue
            key = node_key(model)
            if key in classes or key == ANY_ENDPOINT_KEY:
                skipped += 1
                logger.debug("Duplicate dynamic node key skipped: %s", key)
                continue
            node_class = build_node_class(model)
            endpoint_id = str(model.get("endpoint_id") or "")

            preferred = build_display_name(model)
            if endpoint_id in featured:
                category = str(model.get("category") or "other")
                node_class.CATEGORY = f"FAL/Featured/{category}"
                preferred = featured[endpoint_id] or preferred
            if endpoint_id in superseded:
                newest_id, newest_date = superseded[endpoint_id]
                _apply_superseded_note(node_class, newest_id, newest_date)
                flagged += 1

            name = _unique_display_name(preferred, used_names)
            classes = {**classes, key: node_class}
            display = {**display, key: name}
            used_names.add(name)
        except Exception as err:
            skipped += 1
            logger.debug(
                "Skipped dynamic node for %s: %s",
                model.get("endpoint_id", "<unknown>"),
                err,
            )

    return classes, display, skipped, flagged


def _log_missing_featured(featured: dict[str, str | None], models: list[dict[str, Any]]) -> int:
    """Debug-log featured ids absent from the registry; returns how many matched."""
    registry_ids = {str(m.get("endpoint_id") or "") for m in models}
    missing = [endpoint_id for endpoint_id in featured if endpoint_id not in registry_ids]
    for endpoint_id in missing:
        logger.debug("Featured model not in registry, skipped: %s", endpoint_id)
    return len(featured) - len(missing)


def _schedule_freshness_check() -> None:
    """Kick off the delayed registry freshness check; never raises."""
    try:
        from ..utils.freshness import schedule_startup_check

        schedule_startup_check()
    except Exception as err:
        logger.debug("Could not schedule registry freshness check: %s", err)


def load_dynamic_mappings() -> Mappings:
    """Build (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS) for dynamic nodes."""
    try:
        if not _truthy(_get_setting("dynamic_nodes", "enabled", True)):
            logger.info("Dynamic fal nodes disabled via config")
            return {}, {}

        categories = _category_filter()
        models = _read_models()
        featured = _read_featured()
        featured_count = _log_missing_featured(featured, models)
        superseded = _superseded_map(models)
        classes, display, skipped, flagged = _build_model_mappings(
            models, categories, featured=featured, superseded=superseded
        )

        all_classes = {ANY_ENDPOINT_KEY: FalAnyEndpoint, **classes}
        all_display = {ANY_ENDPOINT_KEY: ANY_ENDPOINT_DISPLAY_NAME, **display}

        logger.info(
            "Registered %d dynamic fal nodes (skipped %d, featured %d, "
            "%d flagged as superseded within their family)",
            len(all_classes),
            skipped,
            featured_count,
            flagged,
        )
        _schedule_freshness_check()
        return all_classes, all_display
    except Exception as err:
        logger.error("Dynamic fal node loading failed entirely: %s", err)
        return {}, {}
