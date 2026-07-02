"""Return-type specs per output kind and result post-processing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..fal_utils import FalApiError, MediaUtils, ResultProcessor

RETURN_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "images": (("IMAGE",), ("images",)),
    "image": (("IMAGE",), ("images",)),
    "video": (("VIDEO", "STRING"), ("video", "video_url")),
    "audio": (("AUDIO", "STRING"), ("audio", "audio_url")),
    "text": (("STRING",), ("text",)),
    "file": (("STRING",), ("file_url",)),
    "json": (("STRING",), ("json",)),
}

_FILE_PROP_CANDIDATES = (
    "model_glb",
    "model_mesh",
    "model_url",
    "model_urls",
    "file",
    "file_url",
    "output",
    "outputs",
)


def find_url(value: Any) -> str | None:
    """Recursively dig a result fragment for a URL string."""
    if isinstance(value, str):
        return value if value.startswith(("http://", "https://", "data:")) else None
    if isinstance(value, dict):
        direct = value.get("url")
        if isinstance(direct, str):
            return direct
        for nested in value.values():
            found = find_url(nested)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = find_url(item)
            if found is not None:
                return found
    return None


def _url_from_props(result: dict[str, Any], props: Sequence[str]) -> str | None:
    for prop in props:
        if prop in result:
            found = find_url(result[prop])
            if found is not None:
                return found
    return None


def _media_url(model: dict[str, Any], result: dict[str, Any], primary: str) -> str:
    props: list[str] = [primary]
    for prop in model.get("output_props") or []:
        if prop not in props:
            props = [*props, prop]
    url = _url_from_props(result, props)
    if url is None:
        url = find_url(result)
    if url is None:
        raise FalApiError(
            model["endpoint_id"], f"No {primary} URL found in API result"
        )
    return url


def _process_video(model: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    url = _media_url(model, result, "video")
    return (MediaUtils.video_from_url(url), url)


def _process_audio(model: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    url = _media_url(model, result, "audio")
    return (MediaUtils.audio_from_url(url), url)


def _process_text(model: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    for prop in model.get("output_props") or []:
        value = result.get(prop)
        if isinstance(value, str):
            return (value,)
    for value in result.values():
        if isinstance(value, str):
            return (value,)
    return (json.dumps(result, default=str),)


def _process_file(model: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    props = [*(model.get("output_props") or []), *_FILE_PROP_CANDIDATES]
    url = _url_from_props(result, props)
    if url is None:
        url = find_url(result)
    if url is None:
        raise FalApiError(model["endpoint_id"], "No file URL found in API result")
    return (url,)


def process_result(model: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    """Convert a raw fal API result dict into the node's return tuple."""
    kind = model.get("output_kind", "json")
    if kind == "images":
        return ResultProcessor.process_image_result(result)
    if kind == "image":
        return ResultProcessor.process_single_image_result(result)
    if kind == "video":
        return _process_video(model, result)
    if kind == "audio":
        return _process_audio(model, result)
    if kind == "text":
        return _process_text(model, result)
    if kind == "file":
        return _process_file(model, result)
    return (json.dumps(result, default=str),)
