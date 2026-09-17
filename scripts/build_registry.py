#!/usr/bin/env python3
"""Build a compact registry of fal.ai model endpoints.

Distills the fal.ai model catalog plus per-endpoint OpenAPI schemas into a
single registry JSON (``data/fal_registry.json``) that a node factory can use
to auto-generate ComfyUI nodes.

Stdlib only. Usage:

    python scripts/build_registry.py \
        --out data/fal_registry.json \
        --since-days 0 \
        --catalog-cache /path/to/fal_models_all.json \
        --schemas-cache /path/to/fal_schemas_recent.json
"""

import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

CATALOG_URL = "https://fal.ai/api/models?page={page}&total=100"
SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={endpoint_id}"
USER_AGENT = "ComfyUI-fal-API-registry-builder/1.0"

FETCH_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.5
MAX_DESCRIPTION_CHARS = 500
MULTILINE_NAMES = frozenset({"prompt", "negative_prompt", "text", "script", "dialogue"})
MULTILINE_DESCRIPTION_THRESHOLD = 120
SKIPPED_PROPERTY_NAMES = frozenset({"sync_mode"})
FILE_OUTPUT_PROPS = frozenset({"model_glb", "model_mesh", "model_url", "model_urls", "mesh"})

logger = logging.getLogger("build_registry")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_json(url):
    """Fetch a URL and parse JSON, with retries and backoff.

    Returns the parsed document, or None for a 404 (skip-and-log).
    Raises on persistent non-404 failure.
    """
    last_error = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                logger.warning("404 for %s, skipping", url)
                return None
            last_error = error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
        time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url} after {FETCH_ATTEMPTS} attempts: {last_error}")


def extract_catalog_items(payload):
    """Normalize a catalog API response page into a list of items."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "models", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def fetch_catalog():
    """Fetch all catalog pages until an empty page is returned."""
    items = []
    page = 1
    while True:
        payload = fetch_json(CATALOG_URL.format(page=page))
        page_items = extract_catalog_items(payload)
        if not page_items:
            break
        items = items + page_items
        logger.info("Fetched catalog page %d (%d items)", page, len(page_items))
        page += 1
    return items


def fetch_schemas(endpoint_ids, max_workers):
    """Fetch OpenAPI docs for endpoint ids concurrently. Returns id -> doc."""
    schemas = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_json, SCHEMA_URL.format(endpoint_id=endpoint_id)): endpoint_id
            for endpoint_id in endpoint_ids
        }
        for future in as_completed(futures):
            endpoint_id = futures[future]
            try:
                doc = future.result()
            except RuntimeError as error:
                logger.warning("Schema fetch failed for %s: %s", endpoint_id, error)
                continue
            if doc is not None:
                schemas = {**schemas, endpoint_id: doc}
    return schemas


# ---------------------------------------------------------------------------
# Catalog filtering
# ---------------------------------------------------------------------------

def parse_published_at(item):
    """Parse the model's publication timestamp, or None."""
    raw = item.get("publishedAt") or item.get("date") or ""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_catalog(catalog, since):
    """Keep live, public models (optionally within a publish window; since=None keeps all).

    Returns (kept_items, skip_reason_counter).
    """
    kept = []
    skipped = Counter()
    seen_ids = set()
    for item in catalog:
        endpoint_id = item.get("id") or ""
        if not endpoint_id or endpoint_id in seen_ids:
            skipped["duplicate_or_missing_id"] += 1
            continue
        seen_ids.add(endpoint_id)
        if item.get("status") != "public":
            skipped["not_public"] += 1
            continue
        if item.get("deprecated"):
            skipped["deprecated"] += 1
            continue
        if item.get("removed"):
            skipped["removed"] += 1
            continue
        if since is not None:
            published = parse_published_at(item)
            if published is None or published < since:
                skipped["outside_window"] += 1
                continue
        kept = kept + [item]
    return kept, skipped


# ---------------------------------------------------------------------------
# Schema resolution helpers
# ---------------------------------------------------------------------------

def resolve_ref(schema, components):
    """Resolve a local $ref against components.schemas, one level."""
    ref = schema.get("$ref", "")
    if not ref.startswith("#/components/schemas/"):
        return schema
    name = ref.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    resolved = components.get(name)
    if not isinstance(resolved, dict):
        return schema
    siblings = {key: value for key, value in schema.items() if key != "$ref"}
    return {**resolved, **siblings}


def is_custom_size_pair(branches):
    """Detect the image_size pattern: [enum-of-presets, width/height object]."""
    enum_branch = next((b for b in branches if b.get("enum")), None)
    object_branch = next(
        (
            b
            for b in branches
            if b.get("type") == "object" or "properties" in b
        ),
        None,
    )
    if enum_branch is None or object_branch is None:
        return None
    properties = object_branch.get("properties", {})
    if "width" in properties and "height" in properties:
        return enum_branch
    return None


def normalize_schema(schema, components, _seen_refs=frozenset()):
    """Resolve nested references and composition without dropping enum choices.

    Returns (resolved_schema, has_custom_size, custom_size_enum_values).
    """
    if not isinstance(schema, dict):
        return {}, False, None
    ref = schema.get("$ref")
    if ref and ref in _seen_refs:
        return {}, False, None
    seen = _seen_refs | {ref} if ref else _seen_refs
    resolved = resolve_ref(schema, components)
    if "$ref" in resolved and resolved != schema:
        return normalize_schema(resolved, components, seen)
    has_custom_size, custom_values = False, None
    if "allOf" in resolved:
        merged = {}
        for branch in resolved["allOf"]:
            normalized, branch_custom, branch_values = normalize_schema(branch, components, seen)
            if branch_custom:
                has_custom_size, custom_values = True, branch_values
            properties = {**merged.get("properties", {}), **normalized.get("properties", {})}
            required = list(dict.fromkeys(merged.get("required", []) + normalized.get("required", [])))
            if "enum" in merged and "enum" in normalized:
                normalized = {**normalized, "enum": [v for v in merged["enum"] if v in normalized["enum"]]}
            merged = {**merged, **normalized}
            if properties:
                merged["properties"] = properties
            if required:
                merged["required"] = required
        siblings = {key: value for key, value in resolved.items() if key != "allOf"}
        if "properties" in siblings:
            siblings["properties"] = {**merged.get("properties", {}), **siblings["properties"]}
        if "required" in siblings:
            siblings["required"] = list(dict.fromkeys(merged.get("required", []) + siblings["required"]))
        resolved = {**merged, **siblings}
    if "const" in resolved:
        literal = resolved["const"]
        resolved = {key: value for key, value in resolved.items() if key != "const"}
        if literal is None:
            resolved = {**resolved, "type": "null"}
        else:
            resolved = {**resolved, "enum": [literal]}
    if isinstance(resolved.get("type"), list):
        types = [value for value in resolved["type"] if value != "null"]
        if len(types) == 1:
            resolved = {**resolved, "type": types[0]}
    branches_key = "anyOf" if "anyOf" in resolved else ("oneOf" if "oneOf" in resolved else None)
    if branches_key is None:
        return resolved, has_custom_size, custom_values

    normalized_branches = [normalize_schema(branch, components, seen) for branch in resolved[branches_key]]
    normalized_branches = [entry for entry in normalized_branches if entry[0] and entry[0].get("type") != "null"]
    branches = [entry[0] for entry in normalized_branches]
    siblings = {key: value for key, value in resolved.items() if key != branches_key}
    if not branches:
        return siblings, False, None
    if len(branches) == 1:
        branch, branch_custom, branch_values = normalized_branches[0]
        return {**branch, **siblings}, branch_custom, branch_values

    custom_enum_branch = is_custom_size_pair(branches)
    if custom_enum_branch is not None:
        values = list(custom_enum_branch.get("enum", [])) + ["custom_size"]
        return {**custom_enum_branch, **siblings}, True, values

    if all(branch.get("enum") for branch in branches):
        values = []
        for branch in branches:
            for value in branch["enum"]:
                if value is not None and value not in values:
                    values.append(value)
        if "enum" in siblings:
            values = [value for value in values if value in siblings["enum"]]
        return {**branches[0], **siblings, "enum": values}, False, None

    # An enum plus an open string branch is still an open string. Keep the
    # literals as suggestions instead of incorrectly restricting API values.
    open_string = next((b for b in branches if b.get("type") == "string" and not b.get("enum")), None)
    if open_string is not None and all(
        b.get("type") == "string" or (b.get("enum") and all(isinstance(value, str) for value in b["enum"]))
        for b in branches
    ):
        examples = [value for b in branches for value in b.get("enum", b.get("examples", []))]
        return {**open_string, "examples": examples, **siblings}, False, None

    enum_branch = next((branch for branch in branches if branch.get("enum")), None)
    chosen = enum_branch if enum_branch is not None else branches[0]
    return {**chosen, **siblings}, False, None


# ---------------------------------------------------------------------------
# Input distillation
# ---------------------------------------------------------------------------

def detect_media_kind(name, schema, is_list):
    """Heuristic media kind from a property name (string-typed props only)."""
    lowered = name.lower()
    description = str(schema.get("description", "")).lower()
    if "image_url" in lowered or "mask_url" in lowered:
        return "image"
    if lowered.endswith("_image"):
        return "image"
    if "video_url" in lowered:
        return "video"
    if "audio_url" in lowered or "voice_url" in lowered:
        return "audio"
    if "_url" in lowered or lowered == "url" or schema.get("format") == "uri":
        for kind in ("image", "video", "audio"):
            if kind in description:
                return kind
        return "file"
    del is_list  # signature symmetry; list-ness does not change the kind
    return None


def trim_text(value, limit=MAX_DESCRIPTION_CHARS):
    """Trim a description/title string."""
    return str(value or "").strip()[:limit]


def scalar_type_of(schema):
    """Map an OpenAPI scalar type to a registry type."""
    type_name = schema.get("type")
    if schema.get("enum"):
        return "enum"
    if type_name in ("integer", "number", "boolean", "string"):
        return type_name
    if type_name == "object" or "properties" in schema:
        return "json"
    return "json"


def string_suggestions(name, schema):
    """Short example identifiers are suggestions, never strict enum constraints.

    Keep prose, prompts and formatted strings as text. The frontend offers
    custom values so undocumented languages, voices, model IDs and
    future modes remain usable even when examples are incomplete.
    """
    if name in MULTILINE_NAMES or name.endswith(("_prompt", "_text")) or schema.get("format"):
        return None
    examples = schema.get("examples")
    if not isinstance(examples, list) or not all(
        isinstance(value, str) and re.fullmatch(r"[\w./:+-]{1,80}", value) and "://" not in value
        for value in examples
    ):
        return None
    values = list(dict.fromkeys(examples))
    if len(values) < 2:
        return None
    return values


def distill_property(name, raw_schema, required_names, components):
    """Distill one input property into a registry input record, or None."""
    if name in SKIPPED_PROPERTY_NAMES or name.startswith("_"):
        return None

    schema, has_custom_size, custom_enum = normalize_schema(raw_schema, components)

    is_list = False
    if schema.get("type") == "array":
        is_list = True
        items, _, _ = normalize_schema(schema.get("items", {}), components)
        item_type = scalar_type_of(items)
        if item_type == "json":
            type_name = "json"
            is_list = False  # rendered as a single JSON field
        else:
            type_name = item_type
        item_schema = items
    else:
        type_name = scalar_type_of(schema)
        item_schema = schema

    enum_values = None
    if has_custom_size:
        type_name = "enum"
        enum_values = custom_enum
    elif type_name == "enum":
        enum_values = list(item_schema.get("enum", []))

    minimum = schema.get("minimum", schema.get("exclusiveMinimum"))
    maximum = schema.get("maximum", schema.get("exclusiveMaximum"))
    if type_name not in ("integer", "number"):
        minimum = None
        maximum = None

    default = schema.get("default", raw_schema.get("default") if isinstance(raw_schema, dict) else None)
    if type_name == "json" and default is not None and not isinstance(default, str):
        default = json.dumps(default, ensure_ascii=False, sort_keys=True)

    # Some upstream schemas declare enum members and the default with mismatched
    # types (e.g. enum ["1","2","4","8"] with default 4). Normalize the default
    # onto the literal enum member it string-matches so widgets get a valid value.
    if enum_values and default is not None and default not in enum_values:
        match = next((v for v in enum_values if str(v) == str(default)), None)
        if match is not None:
            default = match

    description = trim_text(schema.get("description") or schema.get("title"))

    media_kind = None
    if type_name == "string" or (is_list and type_name == "string"):
        media_kind = detect_media_kind(name, schema, is_list)

    multiline = name in MULTILINE_NAMES or (
        type_name == "string"
        and not enum_values
        and len(description) > MULTILINE_DESCRIPTION_THRESHOLD
    )

    record = {
        "name": name,
        "type": type_name,
        "required": name in required_names,
        "default": default,
        "enum": enum_values,
        "min": minimum,
        "max": maximum,
        "description": description,
        "media_kind": media_kind,
        "is_list": is_list,
        "multiline": multiline,
    }
    if has_custom_size:
        record = {**record, "has_custom_size": True}
    if type_name == "string" and not is_list and not media_kind:
        suggestions = string_suggestions(name, schema)
        if suggestions:
            record = {**record, "suggestions": suggestions}
    return record


def ordered_property_names(schema):
    """Property names, preferring fal's declared ordering."""
    properties = schema.get("properties", {})
    declared = schema.get("x-fal-order-properties")
    if isinstance(declared, list):
        ordered = [name for name in declared if name in properties]
        remainder = [name for name in properties if name not in ordered]
        return ordered + remainder
    return list(properties)


def distill_inputs(schema, components, endpoint_id):
    """Distill an Input schema's properties into registry input records."""
    schema, _, _ = normalize_schema(schema, components)
    properties = schema.get("properties", {})
    required_names = set(schema.get("required", []))
    names = ordered_property_names(schema)

    inputs = []
    for name in names:
        record = distill_property(name, properties.get(name, {}), required_names, components)
        if record is not None:
            inputs = inputs + [record]
    return inputs


# ---------------------------------------------------------------------------
# Schema selection
# ---------------------------------------------------------------------------

def ref_name(schema):
    """Extract the local component name from a {'$ref': ...} node."""
    ref = schema.get("$ref", "") if isinstance(schema, dict) else ""
    return ref.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~") if ref.startswith("#/components/schemas/") else None


def input_ref_from_paths(doc):
    """Name of the schema referenced by the app POST requestBody."""
    for operations in doc.get("paths", {}).values():
        post = operations.get("post") if isinstance(operations, dict) else None
        if not isinstance(post, dict):
            continue
        content = post.get("requestBody", {}).get("content", {})
        schema = content.get("application/json", {}).get("schema", {})
        name = ref_name(schema)
        if name:
            return name
    return None


def output_ref_from_paths(doc):
    """Name of the schema referenced by result GET responses."""
    for operations in doc.get("paths", {}).values():
        get = operations.get("get") if isinstance(operations, dict) else None
        if not isinstance(get, dict):
            continue
        for response in get.get("responses", {}).values():
            content = response.get("content", {}) if isinstance(response, dict) else {}
            schema = content.get("application/json", {}).get("schema", {})
            name = ref_name(schema)
            if name and name.endswith("Output"):
                return name
    return None


def select_schema(doc, endpoint_id, suffix, path_lookup):
    """Select the app Input/Output schema from components.schemas."""
    components = doc.get("components", {}).get("schemas", {})
    # A document can describe several sibling endpoints. Prefer this endpoint's
    # operation, including inline schemas, over whichever path happens to be first.
    path = "/" + endpoint_id.strip("/")
    if suffix == "Input":
        operation = doc.get("paths", {}).get(path, {}).get("post", {})
        content = operation.get("requestBody", {}).get("content", {})
    else:
        operation = doc.get("paths", {}).get(path + "/requests/{request_id}", {}).get("get", {})
        content = operation.get("responses", {}).get("200", {}).get("content", {})
    schema = content.get("application/json", {}).get("schema")
    if isinstance(schema, dict):
        return normalize_schema(schema, components)[0]
    referenced = path_lookup(doc)
    if referenced and referenced in components:
        return normalize_schema(components[referenced], components)[0]

    candidates = [name for name in components if name.endswith(suffix)]
    if not candidates:
        return None
    normalized_endpoint = "".join(ch for ch in endpoint_id.lower() if ch.isalnum())
    matching = [
        name
        for name in candidates
        if "".join(ch for ch in name.lower() if ch.isalnum()).replace(suffix.lower(), "")
        in normalized_endpoint
    ]
    pool = matching or candidates
    return normalize_schema(components[max(pool, key=len)], components)[0]


# ---------------------------------------------------------------------------
# Output kind detection
# ---------------------------------------------------------------------------

def detect_output(schema, components):
    """Classify an Output schema. Returns (output_kind, output_props)."""
    if schema is None:
        return "json", []
    properties = schema.get("properties", {})
    prop_names = list(properties)
    lowered = {name.lower() for name in prop_names}

    def prop_is_array(name):
        resolved, _, _ = normalize_schema(properties.get(name, {}), components)
        return resolved.get("type") == "array"

    if "images" in lowered and prop_is_array("images"):
        return "images", prop_names
    if "image" in lowered:
        return "image", prop_names
    if "video" in lowered or "videos" in lowered:
        return "video", prop_names
    if "audio" in lowered or "audios" in lowered:
        return "audio", prop_names
    if lowered & FILE_OUTPUT_PROPS:
        return "file", prop_names

    if prop_names:
        all_stringlike = True
        for name in prop_names:
            resolved, _, _ = normalize_schema(properties.get(name, {}), components)
            if scalar_type_of(resolved) != "string":
                all_stringlike = False
                break
        if all_stringlike:
            return "text", prop_names

    return "json", prop_names


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_record(item, doc):
    """Build a single registry record from a catalog item + OpenAPI doc."""
    endpoint_id = item["id"]
    components = doc.get("components", {}).get("schemas", {})

    input_schema = select_schema(doc, endpoint_id, "Input", input_ref_from_paths)
    if input_schema is None:
        logger.warning("%s: no Input schema found, skipping", endpoint_id)
        return None

    output_schema = select_schema(doc, endpoint_id, "Output", output_ref_from_paths)
    output_kind, output_props = detect_output(output_schema, components)

    published = parse_published_at(item)
    pricing = str(item.get("pricingInfoOverride") or "").replace("**", "").strip()

    return {
        "endpoint_id": endpoint_id,
        "title": str(item.get("title") or "").strip(),
        "category": str(item.get("category") or "").strip(),
        "lab": str(item.get("modelLab") or "").strip(),
        "family": str(item.get("modelFamily") or "").strip(),
        "description": trim_text(item.get("shortDescription")),
        "pricing": pricing,
        "published_at": published.isoformat() if published else "",
        "thumbnail": str(item.get("thumbnailUrl") or "").strip(),
        "inputs": distill_inputs(input_schema, components, endpoint_id),
        "output_kind": output_kind,
        "output_props": output_props,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_json_file(path):
    """Load a JSON cache file."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to load cache file {path}: {error}") from error


def preserve_missing_records(records, baseline):
    """Retain historical endpoints so saved ComfyUI workflows keep loading.

    A missing endpoint may be retired or its OpenAPI schema may be temporarily
    unavailable. The old schema remains usable for node registration and is
    moved to a compatibility tier at runtime. If the endpoint returns in a
    later catalog build, its fresh record automatically replaces this copy.
    """
    current_ids = {record["endpoint_id"] for record in records}
    preserved = []
    for model in baseline.get("models") or []:
        if not isinstance(model, dict) or not model.get("endpoint_id"):
            continue
        if model["endpoint_id"] in current_ids:
            continue
        compatibility_record = {
            **model,
            "deprecated": True,
            "deprecated_reason": "Endpoint absent from the latest live fal catalog or schema fetch.",
        }
        preserved.append(compatibility_record)
    return records + preserved


def parse_args():
    parser = argparse.ArgumentParser(description="Build the fal.ai model registry JSON.")
    parser.add_argument("--out", default="data/fal_registry.json", help="Output registry path")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Rolling publish window in days; 0 (default) = all live models",
    )
    parser.add_argument("--catalog-cache", default=None, help="Path to cached catalog JSON")
    parser.add_argument("--schemas-cache", default=None, help="Path to cached endpoint_id->OpenAPI JSON")
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent schema fetches")
    parser.add_argument(
        "--preserve-from",
        default=None,
        help="Registry whose missing endpoints should be retained for workflow compatibility",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Drop endpoints missing from the live build instead of preserving compatibility nodes",
    )
    return parser.parse_args()


def log_summary(records, skipped):
    """Log counts by category / output kind and skip reasons."""
    category_counts = Counter(record["category"] for record in records)
    kind_counts = Counter(record["output_kind"] for record in records)
    logger.info("Models by category:")
    for category, count in category_counts.most_common():
        logger.info("  %-28s %d", category or "(none)", count)
    logger.info("Models by output_kind:")
    for kind, count in kind_counts.most_common():
        logger.info("  %-10s %d", kind, count)
    logger.info("Skipped: %s", dict(skipped) or "none")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    baseline = None
    baseline_path = args.preserve_from or args.out
    if not args.prune_missing and os.path.isfile(baseline_path):
        baseline = load_json_file(baseline_path)

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.since_days) if args.since_days > 0 else None

    catalog = (
        load_json_file(args.catalog_cache) if args.catalog_cache else fetch_catalog()
    )
    logger.info("Catalog: %d items", len(catalog))

    kept, skipped = filter_catalog(catalog, since)
    logger.info("After filtering: %d live public models in window", len(kept))

    if args.schemas_cache:
        schemas = load_json_file(args.schemas_cache)
    else:
        schemas = fetch_schemas([item["id"] for item in kept], args.max_workers)
    logger.info("Schemas available: %d", len(schemas))

    records = []
    for item in kept:
        doc = schemas.get(item["id"])
        if doc is None:
            skipped["no_schema"] += 1
            logger.warning("%s: no schema available, skipping", item["id"])
            continue
        record = build_record(item, doc)
        if record is None:
            skipped["no_input_schema"] += 1
            continue
        records = records + [record]

    live_model_count = len(records)
    if baseline is not None:
        records = preserve_missing_records(records, baseline)
    records = sorted(records, key=lambda record: record["endpoint_id"])
    deprecated_model_count = sum(bool(record.get("deprecated")) for record in records)

    # NOTE: no wall-clock fields (generated_at etc.) — the committed registry
    # must be content-deterministic so the weekly refresh workflow only opens a
    # PR when the model set actually changes.
    registry = {
        "version": 1,
        "window_days": args.since_days,
        "model_count": len(records),
        "live_model_count": live_model_count,
        "deprecated_model_count": deprecated_model_count,
        "models": records,
    }

    # atomic write: the live sidebar refresh runs this inside a running
    # ComfyUI — a crash mid-write must not corrupt the tracked registry
    tmp_out = args.out + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as handle:
        json.dump(
            registry,
            handle,
            indent=None,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")

    os.replace(tmp_out, args.out)

    log_summary(records, skipped)
    logger.info(
        "Wrote %d models to %s (%d live, %d compatibility-preserved)",
        len(records),
        args.out,
        live_model_count,
        deprecated_model_count,
    )


if __name__ == "__main__":
    main()
