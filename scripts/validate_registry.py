#!/usr/bin/env python3
"""Validate a generated fal model registry before it replaces the baseline.

The scheduled registry refresh uses this dependency-free gate to reject
truncated catalog responses, duplicate or malformed records, nondeterministic
ordering, and unexpectedly large endpoint changes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = {
    "version",
    "models",
    "model_count",
    "live_model_count",
    "deprecated_model_count",
}
REQUIRED_MODEL_FIELDS = {
    "endpoint_id",
    "title",
    "category",
    "description",
    "family",
    "lab",
    "pricing",
    "published_at",
    "inputs",
    "output_kind",
    "output_props",
    "thumbnail",
}
OUTPUT_KINDS = {"audio", "file", "image", "images", "json", "text", "video"}
ENDPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class RegistryValidationError(ValueError):
    """Raised when a registry cannot safely be promoted."""


def load_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryValidationError(f"{path} must contain a JSON object")
    return payload


def validate_registry(registry: dict[str, Any], *, min_models: int = 500) -> set[str]:
    missing_top = REQUIRED_TOP_LEVEL - registry.keys()
    if missing_top:
        raise RegistryValidationError(f"Registry is missing top-level fields: {sorted(missing_top)}")

    if registry["version"] != 1:
        raise RegistryValidationError(f"Unsupported registry version: {registry['version']!r}")

    models = registry["models"]
    if not isinstance(models, list):
        raise RegistryValidationError("Registry 'models' must be a list")
    if registry["model_count"] != len(models):
        raise RegistryValidationError(f"model_count={registry['model_count']} does not match {len(models)} records")
    deprecated_count = sum(bool(model.get("deprecated")) for model in models if isinstance(model, dict))
    live_count = len(models) - deprecated_count
    if registry["live_model_count"] != live_count:
        raise RegistryValidationError(
            f"live_model_count={registry['live_model_count']} does not match {live_count} records"
        )
    if registry["deprecated_model_count"] != deprecated_count:
        raise RegistryValidationError(
            f"deprecated_model_count={registry['deprecated_model_count']} does not match "
            f"{deprecated_count} records"
        )
    if len(models) < min_models:
        raise RegistryValidationError(f"Registry has only {len(models)} models; minimum is {min_models}")

    endpoint_ids: list[str] = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise RegistryValidationError(f"Model {index} is not an object")
        missing = REQUIRED_MODEL_FIELDS - model.keys()
        if missing:
            raise RegistryValidationError(f"Model {index} is missing fields: {sorted(missing)}")
        endpoint_id = model["endpoint_id"]
        if (
            not isinstance(endpoint_id, str)
            or not endpoint_id.strip()
            or not ENDPOINT_ID_PATTERN.fullmatch(endpoint_id)
        ):
            raise RegistryValidationError(f"Model {index} has an invalid endpoint_id")
        if not isinstance(model["title"], str) or not model["title"].strip():
            raise RegistryValidationError(f"{endpoint_id} has an invalid title")
        if not isinstance(model["inputs"], list):
            raise RegistryValidationError(f"{endpoint_id} inputs must be a list")
        if not isinstance(model["output_props"], list):
            raise RegistryValidationError(f"{endpoint_id} output_props must be a list")
        if model["output_kind"] not in OUTPUT_KINDS:
            raise RegistryValidationError(f"{endpoint_id} has unknown output_kind={model['output_kind']!r}")
        if "deprecated" in model and not isinstance(model["deprecated"], bool):
            raise RegistryValidationError(f"{endpoint_id} deprecated must be a boolean")
        endpoint_ids.append(endpoint_id)

    if len(endpoint_ids) != len(set(endpoint_ids)):
        duplicates = sorted(endpoint_id for endpoint_id in set(endpoint_ids) if endpoint_ids.count(endpoint_id) > 1)
        raise RegistryValidationError(f"Duplicate endpoint IDs: {duplicates[:10]}")
    if endpoint_ids != sorted(endpoint_ids):
        raise RegistryValidationError("Models must be sorted by endpoint_id")
    return set(endpoint_ids)


def compare_registries(
    baseline_ids: set[str],
    candidate_ids: set[str],
    *,
    max_removal_fraction: float = 0.05,
    max_addition_fraction: float = 0.25,
    allow_large_change: bool = False,
) -> tuple[set[str], set[str]]:
    if not 0 <= max_removal_fraction <= 1:
        raise RegistryValidationError("max_removal_fraction must be between 0 and 1")
    if not 0 <= max_addition_fraction <= 1:
        raise RegistryValidationError("max_addition_fraction must be between 0 and 1")
    added = candidate_ids - baseline_ids
    removed = baseline_ids - candidate_ids
    removal_fraction = len(removed) / len(baseline_ids) if baseline_ids else 0.0
    addition_fraction = len(added) / len(baseline_ids) if baseline_ids else 0.0
    if removal_fraction > max_removal_fraction and not allow_large_change:
        raise RegistryValidationError(
            f"Candidate removes {len(removed)}/{len(baseline_ids)} endpoints "
            f"({removal_fraction:.1%}), above the {max_removal_fraction:.1%} limit"
        )
    if addition_fraction > max_addition_fraction and not allow_large_change:
        raise RegistryValidationError(
            f"Candidate adds {len(added)}/{len(baseline_ids)} endpoints "
            f"({addition_fraction:.1%}), above the {max_addition_fraction:.1%} limit"
        )
    return added, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Generated registry to validate")
    parser.add_argument("--baseline", type=Path, help="Committed registry to compare")
    parser.add_argument("--min-models", type=int, default=500)
    parser.add_argument("--max-removal-fraction", type=float, default=0.05)
    parser.add_argument("--max-addition-fraction", type=float, default=0.25)
    parser.add_argument("--allow-large-change", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_ids = validate_registry(load_registry(args.candidate), min_models=args.min_models)
    added: set[str] = set()
    removed: set[str] = set()
    if args.baseline:
        baseline_ids = validate_registry(load_registry(args.baseline), min_models=args.min_models)
        added, removed = compare_registries(
            baseline_ids,
            candidate_ids,
            max_removal_fraction=args.max_removal_fraction,
            max_addition_fraction=args.max_addition_fraction,
            allow_large_change=args.allow_large_change,
        )
    print(f"Registry valid: {len(candidate_ids)} models " f"(+{len(added)} / -{len(removed)} vs baseline)")


if __name__ == "__main__":
    main()
