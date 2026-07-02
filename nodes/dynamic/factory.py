"""Builds concrete ComfyUI node classes from registry model entries."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..fal_utils import ApiHandler
from .arguments import build_arguments
from .outputs import RETURN_SPECS, process_result
from .schema_to_inputs import build_input_types

NODE_KEY_PREFIX = "FalAPI_"


def node_key(model: dict[str, Any]) -> str:
    return NODE_KEY_PREFIX + model["endpoint_id"].replace("/", "-")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_display_name(model: dict[str, Any]) -> str:
    endpoint_id = model["endpoint_id"]
    title = model.get("title") or endpoint_id
    parts = endpoint_id.split("/")
    remainder = "/".join(parts[1:]) if len(parts) > 1 else endpoint_id
    if not remainder or _slug(remainder) == _slug(title):
        return f"{title} (fal)"
    return f"{title} · {remainder} (fal)"


def _value_fingerprint(value: Any) -> str:
    # torch tensors: repr() summarizes large tensors (edge elements only), so two
    # different images could hash identically — fingerprint the raw bytes instead
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            tensor = value.detach().cpu().contiguous()
            digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            return f"tensor:{tuple(tensor.shape)}:{tensor.dtype}:{digest}"
        except Exception:  # non-numpy-compatible tensor; fall through to repr
            pass
    if isinstance(value, dict):  # e.g. AUDIO dicts carrying a waveform tensor
        return repr(sorted((k, _value_fingerprint(v)) for k, v in value.items()))
    return repr(value)


def stable_hash(kwargs: dict[str, Any]) -> str:
    payload = repr(sorted((key, _value_fingerprint(value)) for key, value in kwargs.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def _class_name(model: dict[str, Any]) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", node_key(model))


def _description(model: dict[str, Any]) -> str:
    description = model.get("description") or ""
    pricing = model.get("pricing") or ""
    if pricing:
        return f"{description}\n\nPricing: {pricing}".strip()
    return description


def build_node_class(model: dict[str, Any]) -> type:
    """Create a ComfyUI node class for a single registry model entry."""
    endpoint_id = model["endpoint_id"]
    kind = model.get("output_kind", "json")
    return_types, return_names = RETURN_SPECS.get(kind, RETURN_SPECS["json"])
    category = model.get("category") or "other"

    def input_types(cls: type) -> dict[str, Any]:
        return build_input_types(model)

    def is_changed(cls: type, **kwargs: Any) -> Any:
        if kwargs.get("force_rerun"):
            return float("nan")
        return stable_hash(kwargs)

    def run(self: Any, **kwargs: Any) -> tuple:
        arguments = build_arguments(model, kwargs)
        result = ApiHandler.submit_and_get_result(endpoint_id, arguments)
        return process_result(model, result)

    attrs = {
        "INPUT_TYPES": classmethod(input_types),
        "IS_CHANGED": classmethod(is_changed),
        "RETURN_TYPES": return_types,
        "RETURN_NAMES": return_names,
        "FUNCTION": "run",
        "CATEGORY": f"FAL/Models/{category}",
        "DESCRIPTION": _description(model),
        "run": run,
        "_FAL_ENDPOINT_ID": endpoint_id,
    }
    return type(_class_name(model), (object,), attrs)
