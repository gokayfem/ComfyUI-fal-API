"""Pure translation of a registry model schema into a ComfyUI INPUT_TYPES dict."""

from __future__ import annotations

import json
from typing import Any

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1

SEED_SPEC = (
    "INT",
    {
        "default": -1,
        "min": -1,
        "max": INT_MAX,
        "control_after_generate": True,
        "tooltip": "-1 = random (fal picks); any other value is sent to the API",
    },
)

FORCE_RERUN_SPEC = (
    "BOOLEAN",
    {"default": False, "tooltip": "Bypass ComfyUI's cache and call the API again"},
)

WIDTH_HEIGHT_OPTS = {"default": 1024, "min": 64, "max": 14142, "step": 8}

_MEDIA_TYPES = {"image": "IMAGE", "video": "VIDEO", "audio": "AUDIO"}

# media kinds that get a "<name>_direct_url" passthrough companion input
# ("file" is excluded: it is already a plain STRING URL widget)
DIRECT_URL_SUFFIX = "_direct_url"
DIRECT_URL_KINDS = ("image", "video", "audio")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _with_tooltip(opts: dict[str, Any], description: str | None) -> dict[str, Any]:
    if description:
        return {**opts, "tooltip": description}
    return opts


def _int_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    lo = int(inp["min"]) if inp.get("min") is not None else INT_MIN
    hi = int(inp["max"]) if inp.get("max") is not None else INT_MAX
    raw_default = inp.get("default")
    default = int(raw_default) if isinstance(raw_default, (int, float)) else 0
    opts = {"default": int(_clamp(default, lo, hi)), "min": lo, "max": hi}
    return ("INT", _with_tooltip(opts, inp.get("description")))


def _float_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    has_range = inp.get("min") is not None and inp.get("max") is not None
    lo = float(inp["min"]) if inp.get("min") is not None else -1e10
    hi = float(inp["max"]) if inp.get("max") is not None else 1e10
    step = 0.01 if has_range and (hi - lo) <= 10 else 0.1
    raw_default = inp.get("default")
    default = float(raw_default) if isinstance(raw_default, (int, float)) else 0.0
    opts = {"default": _clamp(default, lo, hi), "min": lo, "max": hi, "step": step}
    return ("FLOAT", _with_tooltip(opts, inp.get("description")))


def _multi_enum_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    """Array-of-enum inputs: ComfyUI has no multi-select widget, so use a
    comma-separated string validated at call time."""
    values = list(inp.get("enum") or [])
    default = inp.get("default")
    text = ", ".join(str(v) for v in default) if isinstance(default, list) else ""
    description = (inp.get("description") or "").strip()
    tooltip = f"{description} Comma-separated. Options: {', '.join(values)}".strip()
    return ("STRING", {"default": text, "tooltip": tooltip})


def _enum_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    if inp.get("is_list"):
        return _multi_enum_spec(inp)
    values = list(inp.get("enum") or [])
    if not values:
        return _string_spec(inp)
    default = inp.get("default")
    if default not in values:
        # a dict default on a has_custom_size enum means the API defaults to an
        # explicit {width, height}; represent that as the custom_size preset
        if isinstance(default, dict) and "custom_size" in values:
            default = "custom_size"
        else:
            default = values[0]
    opts = _with_tooltip({"default": default}, inp.get("description"))
    return (values, opts)


def _custom_size_default(inp: dict[str, Any], dimension: str) -> int:
    default = inp.get("default")
    if isinstance(default, dict):
        value = default.get(dimension)
        if isinstance(value, int) and value > 0:
            return int(_clamp(value, 64, 14142))
    return WIDTH_HEIGHT_OPTS["default"]


def _bool_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    opts = {"default": bool(inp.get("default"))}
    return ("BOOLEAN", _with_tooltip(opts, inp.get("description")))


def _string_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    default = inp.get("default")
    opts = {
        "default": default if isinstance(default, str) else "",
        "multiline": bool(inp.get("multiline")),
    }
    return ("STRING", _with_tooltip(opts, inp.get("description")))


def _json_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    default = inp.get("default")
    if default is None:
        text = ""
    elif isinstance(default, str):
        text = default
    else:
        text = json.dumps(default)
    description = (inp.get("description") or "").strip()
    tooltip = (description + " (JSON)").strip()
    return ("STRING", {"default": text, "multiline": True, "tooltip": tooltip})


def _media_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    media_kind = inp.get("media_kind")
    comfy_type = _MEDIA_TYPES.get(media_kind)
    if comfy_type is not None:
        return (comfy_type,)
    # media_kind == "file": plain URL string
    description = (inp.get("description") or "").strip()
    tooltip = (description + " (URL to file)").strip()
    return ("STRING", {"default": "", "tooltip": tooltip})


def _direct_url_spec(name: str, media_kind: str, is_list: bool) -> tuple[Any, ...]:
    tooltip = (
        f"fal/CDN URL passthrough for '{name}': when set, this URL is sent directly "
        f"and the {media_kind} input is ignored — no download/re-upload. "
        "Chain fal nodes' *_url outputs here."
    )
    if is_list:
        tooltip += " Comma-separate multiple URLs."
    return ("STRING", {"default": "", "tooltip": tooltip})


def _direct_url_twin(
    inp: dict[str, Any], existing_names: set[str]
) -> tuple[str, tuple[Any, ...]] | None:
    """The optional passthrough companion for a media input, or None."""
    media_kind = inp.get("media_kind")
    if media_kind not in DIRECT_URL_KINDS:
        return None
    twin_name = inp["name"] + DIRECT_URL_SUFFIX
    if twin_name in existing_names:
        return None
    spec = _direct_url_spec(inp["name"], media_kind, bool(inp.get("is_list")))
    return (twin_name, spec)


def _input_spec(inp: dict[str, Any]) -> tuple[Any, ...]:
    if inp.get("media_kind"):
        return _media_spec(inp)
    input_type = inp.get("type")
    if input_type == "enum":
        return _enum_spec(inp)
    if input_type == "integer":
        return _int_spec(inp)
    if input_type == "number":
        return _float_spec(inp)
    if input_type == "boolean":
        return _bool_spec(inp)
    if input_type in ("json", "object", "array"):
        return _json_spec(inp)
    return _string_spec(inp)


def build_input_types(model: dict[str, Any]) -> dict[str, Any]:
    """Build a ComfyUI INPUT_TYPES dict from a registry model entry."""
    required: dict[str, Any] = {}
    optional: dict[str, Any] = {}
    # twins of *required* media inputs go at the start of the optional bucket
    leading_optional: dict[str, Any] = {}
    custom_size_input: dict[str, Any] | None = None

    inputs = model.get("inputs", [])
    existing_names = {inp["name"] for inp in inputs}

    for inp in inputs:
        name = inp["name"]
        if name == "seed":
            optional[name] = SEED_SPEC
            continue
        if inp.get("has_custom_size") and custom_size_input is None:
            custom_size_input = inp
        spec = _input_spec(inp)
        twin = _direct_url_twin(inp, existing_names)
        if inp.get("required"):
            required[name] = spec
            if twin is not None:
                leading_optional[twin[0]] = twin[1]
        else:
            optional[name] = spec
            if twin is not None:
                optional[twin[0]] = twin[1]

    optional = {**leading_optional, **optional}

    if custom_size_input is not None:
        for dimension in ("width", "height"):
            if dimension not in required and dimension not in optional:
                opts = {
                    **WIDTH_HEIGHT_OPTS,
                    "default": _custom_size_default(custom_size_input, dimension),
                }
                optional[dimension] = ("INT", opts)

    optional["force_rerun"] = FORCE_RERUN_SPEC

    return {"required": required, "optional": optional}
