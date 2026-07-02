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

Mappings = tuple[dict[str, type], dict[str, str]]


def _registry_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    real = package_dir.parents[1] / "data" / _REGISTRY_FILENAME
    if real.is_file():
        return real
    return package_dir / _FIXTURE_FILENAME


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


def _unique_display_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    counter = 2
    while f"{name} #{counter}" in used:
        counter += 1
    return f"{name} #{counter}"


def _build_model_mappings(
    models: list[dict[str, Any]], categories: set[str]
) -> tuple[dict[str, type], dict[str, str], int]:
    classes: dict[str, type] = {}
    display: dict[str, str] = {}
    used_names: set[str] = {ANY_ENDPOINT_DISPLAY_NAME}
    skipped = 0

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
            name = _unique_display_name(build_display_name(model), used_names)
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

    return classes, display, skipped


def load_dynamic_mappings() -> Mappings:
    """Build (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS) for dynamic nodes."""
    try:
        if not _truthy(_get_setting("dynamic_nodes", "enabled", True)):
            logger.info("Dynamic fal nodes disabled via config")
            return {}, {}

        categories = _category_filter()
        models = _read_models()
        classes, display, skipped = _build_model_mappings(models, categories)

        all_classes = {ANY_ENDPOINT_KEY: FalAnyEndpoint, **classes}
        all_display = {ANY_ENDPOINT_KEY: ANY_ENDPOINT_DISPLAY_NAME, **display}

        logger.info(
            "Registered %d dynamic fal nodes (skipped %d)", len(all_classes), skipped
        )
        return all_classes, all_display
    except Exception as err:
        logger.error("Dynamic fal node loading failed entirely: %s", err)
        return {}, {}
