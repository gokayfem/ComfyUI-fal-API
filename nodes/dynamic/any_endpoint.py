"""Generic node that calls any fal.ai endpoint by id with free-form JSON arguments."""

from __future__ import annotations

import json
from typing import Any

from ..fal_utils import (
    ApiHandler,
    FalApiError,
    ImageUtils,
    MediaUtils,
    ResultProcessor,
    logger,
)
from .factory import stable_hash
from .outputs import find_url

ANY_ENDPOINT_KEY = "FalAnyEndpoint_fal"
ANY_ENDPOINT_DISPLAY_NAME = "Fal Any Endpoint (fal)"


def _parse_arguments_json(endpoint_id: str, arguments_json: str) -> dict[str, Any]:
    text = (arguments_json or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except ValueError as err:
        raise FalApiError(endpoint_id, f"Invalid JSON in 'arguments_json': {err}") from err
    if not isinstance(parsed, dict):
        raise FalApiError(endpoint_id, "'arguments_json' must be a JSON object")
    return parsed


def _media_overlay(
    image: Any, image_2: Any, video: Any, audio: Any, seed: int
) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if image is not None:
        first_url = ImageUtils.upload_image(image)
        overlay = {**overlay, "image_url": first_url}
        if image_2 is not None:
            second_url = ImageUtils.upload_image(image_2)
            overlay = {**overlay, "image_urls": [first_url, second_url]}
    if video is not None:
        overlay = {**overlay, "video_url": MediaUtils.upload_video(video)}
    if audio is not None:
        overlay = {**overlay, "audio_url": MediaUtils.upload_audio(audio)}
    if int(seed) != -1:
        overlay = {**overlay, "seed": int(seed)}
    return overlay


def build_overlay_arguments(
    endpoint_id: str,
    arguments_json: str,
    image: Any = None,
    image_2: Any = None,
    video: Any = None,
    audio: Any = None,
    seed: int = -1,
) -> dict[str, Any]:
    """Merge free-form JSON arguments with uploaded media inputs and seed.

    Connected media inputs win over matching keys in the JSON
    (image_url, image_urls, video_url, audio_url, seed).
    """
    parsed = _parse_arguments_json(endpoint_id, arguments_json)
    overlay = _media_overlay(image, image_2, video, audio, seed)
    return {**parsed, **overlay}


def _extract_images(result: dict[str, Any]) -> Any | None:
    try:
        images = result.get("images")
        if isinstance(images, list) and images:
            return ResultProcessor.process_image_result(result)[0]
        if isinstance(result.get("image"), dict):
            return ResultProcessor.process_single_image_result(result)[0]
    except Exception as err:
        logger.debug("FalAnyEndpoint: could not extract images: %s", err)
    return None


def _extract_video(result: dict[str, Any]) -> Any | None:
    try:
        url = find_url(result.get("video"))
        if url is not None:
            return MediaUtils.video_from_url(url)
    except Exception as err:
        logger.debug("FalAnyEndpoint: could not extract video: %s", err)
    return None


def _extract_audio(result: dict[str, Any]) -> Any | None:
    try:
        url = find_url(result.get("audio"))
        if url is not None:
            return MediaUtils.audio_from_url(url)
    except Exception as err:
        logger.debug("FalAnyEndpoint: could not extract audio: %s", err)
    return None


def extract_flexible_outputs(result: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    """Opportunistically extract (images, video, audio, raw json) from a result.

    Each media slot is None when the result has no matching content; the raw
    result is always available as a JSON string in the last slot.
    """
    return (
        _extract_images(result),
        _extract_video(result),
        _extract_audio(result),
        json.dumps(result, default=str),
    )


class FalAnyEndpoint:
    """Call any fal.ai endpoint with raw JSON arguments plus optional media inputs."""

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "video", "audio", "result_json")
    FUNCTION = "run"
    CATEGORY = "FAL/Models"
    DESCRIPTION = (
        "Call any fal.ai endpoint by id. Provide arguments as a JSON object; "
        "connected media inputs are uploaded and override matching keys "
        "(image_url, image_urls, video_url, audio_url, seed) in the JSON. "
        "Outputs are extracted opportunistically; the raw result is always "
        "available as JSON."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "endpoint_id": (
                    "STRING",
                    {
                        "default": "fal-ai/flux/dev",
                        "tooltip": "fal endpoint id, e.g. fal-ai/flux/dev",
                    },
                ),
                "arguments_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": (
                            "JSON object of API arguments. Connected media inputs "
                            "and seed override matching keys here."
                        ),
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Uploaded and sent as image_url"}),
                "image_2": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Second image; when set together with 'image', both are "
                            "also sent as image_urls [url1, url2]"
                        )
                    },
                ),
                "video": ("VIDEO", {"tooltip": "Uploaded and sent as video_url"}),
                "audio": ("AUDIO", {"tooltip": "Uploaded and sent as audio_url"}),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "control_after_generate": True,
                        "tooltip": "-1 = omit seed; any other value is sent to the API",
                    },
                ),
                "force_rerun": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Bypass ComfyUI's cache and call the API again",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        if kwargs.get("force_rerun"):
            return float("nan")
        return stable_hash(kwargs)

    def run(
        self,
        endpoint_id: str,
        arguments_json: str = "{}",
        image: Any = None,
        image_2: Any = None,
        video: Any = None,
        audio: Any = None,
        seed: int = -1,
        force_rerun: bool = False,
    ) -> tuple[Any, Any, Any, str]:
        endpoint = (endpoint_id or "").strip()
        if not endpoint:
            raise FalApiError("(any endpoint)", "endpoint_id is required")

        arguments = build_overlay_arguments(
            endpoint, arguments_json, image, image_2, video, audio, seed
        )

        result = ApiHandler.submit_and_get_result(
            endpoint, arguments, skip_cache=bool(force_rerun)
        )

        return extract_flexible_outputs(result)
