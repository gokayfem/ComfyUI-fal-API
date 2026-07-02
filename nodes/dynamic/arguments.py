"""Pure translation of ComfyUI node kwargs back into fal API arguments."""

from __future__ import annotations

import json
from typing import Any

from ..fal_utils import FalApiError, ImageUtils, MediaUtils
from .schema_to_inputs import DIRECT_URL_KINDS, DIRECT_URL_SUFFIX

_DEFAULT_DIMENSION = 1024


def _direct_url_value(inp: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """The stripped '<name>_direct_url' kwarg for a media input, or ''."""
    if inp.get("media_kind") not in DIRECT_URL_KINDS:
        return ""
    raw = kwargs.get(inp["name"] + DIRECT_URL_SUFFIX)
    return str(raw).strip() if isinstance(raw, str) else ""


def _direct_url_argument(endpoint: str, inp: dict[str, Any], text: str) -> Any:
    """Validate a passthrough URL string; is_list inputs accept comma-separated URLs."""
    twin_name = inp["name"] + DIRECT_URL_SUFFIX
    error = FalApiError(endpoint, f"'{twin_name}' must be an http(s) URL")
    if not inp.get("is_list"):
        if not text.startswith(("http://", "https://")):
            raise error
        return text
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts or any(not part.startswith(("http://", "https://")) for part in parts):
        raise error
    return parts


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
    inputs = model.get("inputs", [])
    input_names = {inp["name"] for inp in inputs}

    for inp in inputs:
        name = inp["name"]
        # URL passthrough wins over the media input (which may be None or connected);
        # skip when the twin name is a real model input (no twin was generated then)
        if name + DIRECT_URL_SUFFIX not in input_names:
            direct_url = _direct_url_value(inp, kwargs)
            if direct_url:
                resolved = _direct_url_argument(endpoint, inp, direct_url)
                arguments = {**arguments, name: resolved}
                continue
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
