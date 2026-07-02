"""Pure translation of ComfyUI node kwargs back into fal API arguments."""

from __future__ import annotations

import json
from typing import Any

from ..fal_utils import FalApiError, ImageUtils, MediaUtils

_DEFAULT_DIMENSION = 1024


def _upload_image(inp: dict[str, Any], value: Any) -> Any:
    if inp.get("is_list"):
        return ImageUtils.prepare_images(value)
    return ImageUtils.upload_image(value)


def _media_argument(inp: dict[str, Any], value: Any) -> Any | None:
    media_kind = inp.get("media_kind")
    if media_kind == "image":
        return _upload_image(inp, value)
    if media_kind == "video":
        return MediaUtils.upload_video(value)
    if media_kind == "audio":
        return MediaUtils.upload_audio(value)
    # media_kind == "file": already a URL string in the widget
    text = str(value).strip()
    return text or None


def _json_argument(endpoint: str, name: str, value: Any) -> Any | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as err:
        raise FalApiError(endpoint, f"Invalid JSON in '{name}': {err}") from err


def _multi_enum_argument(endpoint: str, inp: dict[str, Any], value: Any) -> Any | None:
    """Comma-separated string widget → validated list of enum members."""
    selected = [part.strip() for part in str(value).split(",") if part.strip()]
    if not selected:
        return None
    allowed = set(inp.get("enum") or [])
    invalid = [part for part in selected if part not in allowed]
    if invalid:
        raise FalApiError(
            endpoint,
            f"Invalid value(s) {invalid} for '{inp['name']}'. "
            f"Allowed: {', '.join(sorted(allowed))}",
        )
    return selected


def _enum_argument(inp: dict[str, Any], value: Any, kwargs: dict[str, Any]) -> Any:
    if inp.get("has_custom_size") and value == "custom_size":
        return {
            "width": int(kwargs.get("width", _DEFAULT_DIMENSION)),
            "height": int(kwargs.get("height", _DEFAULT_DIMENSION)),
        }
    return value


def _scalar_argument(
    endpoint: str, inp: dict[str, Any], value: Any, kwargs: dict[str, Any]
) -> Any | None:
    input_type = inp.get("type")
    if input_type == "enum":
        if inp.get("is_list"):
            return _multi_enum_argument(endpoint, inp, value)
        return _enum_argument(inp, value, kwargs)
    if input_type in ("json", "object", "array"):
        return _json_argument(endpoint, inp["name"], value)
    if input_type == "integer":
        return int(value)
    if input_type == "number":
        return float(value)
    if input_type == "boolean":
        return bool(value)
    if input_type == "string":
        if not inp.get("required") and value == "":
            return None
        return value
    return value


def _argument_for(
    endpoint: str, inp: dict[str, Any], value: Any, kwargs: dict[str, Any]
) -> Any | None:
    if inp.get("media_kind"):
        return _media_argument(inp, value)
    return _scalar_argument(endpoint, inp, value, kwargs)


def build_arguments(model: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build the fal API argument dict from node kwargs. Never mutates inputs."""
    endpoint = model["endpoint_id"]
    arguments: dict[str, Any] = {}

    for inp in model.get("inputs", []):
        name = inp["name"]
        if name not in kwargs:
            continue
        value = kwargs[name]
        if value is None:
            continue
        if name == "seed":
            seed = int(value)
            if seed != -1:
                arguments = {**arguments, "seed": seed}
            continue
        resolved = _argument_for(endpoint, inp, value, kwargs)
        if resolved is None:
            continue
        arguments = {**arguments, name: resolved}

    return arguments
