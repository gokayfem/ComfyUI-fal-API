"""Chainable typed builder nodes for JSON inputs on auto-generated fal nodes.

Auto-generated endpoint nodes render complex object/array inputs (registry
type "json") as raw JSON string widgets. The builders here emit exactly the
JSON those fields expect, and each accepts an optional ``chain`` input so N
builders can be daisy-chained to produce an N-element array (or a merged
object for ``FalKeyValue``).

Shapes were validated against the live OpenAPI schemas
(https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>):

- ``LoraWeight``      {path, scale[, weight_name]}         fal-ai/flux-lora,
  fal-ai/wan/v2.2-a14b/text-to-video/lora (126 "loras" inputs in registry)
- ``Embedding``       {path, tokens[]}                      fal-ai/fast-lightning-sdxl
- ``ControlNet``      {path, control_image_url, conditioning_scale,
  start_percentage, end_percentage[, variant]}              fal-ai/flux-general
- ``IPAdapter``       {path, image_encoder_path, image_url, scale
  [, weight_name]}                                          fal-ai/flux-general
- ``ElementInput``    {frontal_image_url, reference_image_urls[]}
  fal-ai/kling-image/o1, fal-ai/kling-image/o3/*
- ``KlingV3MultiPromptElement`` {prompt, duration("1".."15")}
  fal-ai/kling-video/o3/*/image-to-video
"""

from __future__ import annotations

import json
import math
from typing import Any

from .fal_utils import FalApiError, ImageUtils, logger

_CATEGORY = "FAL/Utils/Builders"

_CHAIN_TOOLTIP = (
    "Optional: wire the json output of another builder of the same kind here "
    "to append this entry after its entries (chain N builders for N items)."
)


def _parse_chain(node_name: str, chain: str, container: type) -> Any:
    """Parse a prior chain string into ``container`` (list or dict).

    An empty/blank chain yields a fresh empty container. Anything that is not
    valid JSON of the right container type raises a clear FalApiError.
    """
    text = (chain or "").strip()
    if not text:
        return container()
    try:
        parsed = json.loads(text)
    except ValueError as err:
        logger.error("%s: invalid chain JSON: %s", node_name, err)
        raise FalApiError(node_name, f"'chain' is not valid JSON: {err}") from err
    if not isinstance(parsed, container):
        wanted = "array" if container is list else "object"
        if isinstance(parsed, dict):
            got = "object"
        elif isinstance(parsed, list):
            got = "array"
        else:
            got = type(parsed).__name__
        raise FalApiError(
            node_name,
            f"'chain' must be a JSON {wanted} (got {got}). "
            f"Only chain {node_name}-compatible builders together.",
        )
    return parsed


def _append_entry(node_name: str, chain: str, entry: dict[str, Any]) -> str:
    """New JSON array string: entries from ``chain`` plus ``entry`` (no mutation)."""
    prior = _parse_chain(node_name, chain, list)
    return json.dumps([*prior, entry])


def _require(node_name: str, field: str, value: str) -> str:
    """Strip a required string field, raising when it is blank."""
    text = (value or "").strip()
    if not text:
        raise FalApiError(node_name, f"'{field}' is required and cannot be empty")
    return text


def _resolve_image_url(node_name: str, field: str, image: Any, url: str, required: bool) -> str:
    """A connected IMAGE wins (uploaded via fal storage); else the URL string."""
    if image is not None:
        return ImageUtils.upload_image(image)
    text = (url or "").strip()
    if not text and required:
        raise FalApiError(
            node_name,
            f"Connect an image or fill '{field}': the schema requires an image URL",
        )
    return text


class FalLoRAConfig:
    """Append one LoraWeight ({path, scale}) entry to a JSON array."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build a `loras` JSON array entry ({path, scale}) without hand-writing "
        "JSON. Chain several to stack LoRAs. Wire the json output into the "
        "`loras` field of 126+ fal nodes (fal-ai/flux-lora, "
        "fal-ai/wan/v2.2-a14b/text-to-video/lora, fal-ai/qwen-image, ...)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "URL or Hugging Face id of the LoRA weights, e.g. "
                            "https://.../lora.safetensors. Feeds the `loras` field of "
                            "fal-ai/flux-lora, fal-ai/wan/v2.2-a14b/text-to-video/lora, "
                            "fal-ai/chrono-edit-lora and 120+ more."
                        ),
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "LoRA strength merged into the base model (LoraWeight.scale, 0-4).",
                    },
                ),
            },
            "optional": {
                "weight_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional safetensors file name when `path` is a Hugging Face "
                            "repo with several files (e.g. Wan/Qwen LoRA endpoints). "
                            "Leave empty otherwise."
                        ),
                    },
                ),
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(self, path: str, scale: float, weight_name: str = "", chain: str = "") -> tuple[str]:
        entry: dict[str, Any] = {
            "path": _require("FalLoRAConfig", "path", path),
            "scale": float(scale),
        }
        if (weight_name or "").strip():
            entry = {**entry, "weight_name": weight_name.strip()}
        return (_append_entry("FalLoRAConfig", chain, entry),)


class FalEmbeddingConfig:
    """Append one Embedding ({path, tokens}) entry to a JSON array."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build an `embeddings` JSON array entry ({path, tokens}) for SD/SDXL "
        "endpoints such as fal-ai/fast-lightning-sdxl, fal-ai/dreamshaper and "
        "fal-ai/fast-fooocus-sdxl. Chain several to load multiple embeddings."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "URL or path to the textual-inversion embedding weights, e.g. "
                            "https://civitai.com/api/download/models/135931. Feeds the "
                            "`embeddings` field of fal-ai/fast-lightning-sdxl, "
                            "fal-ai/dreamshaper, fal-ai/fast-fooocus-sdxl."
                        ),
                    },
                ),
            },
            "optional": {
                "tokens": (
                    "STRING",
                    {
                        "default": "<s0>, <s1>",
                        "tooltip": (
                            "Comma-separated trigger tokens for the embedding "
                            "(Embedding.tokens). Leave empty to use the endpoint default."
                        ),
                    },
                ),
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(self, path: str, tokens: str = "<s0>, <s1>", chain: str = "") -> tuple[str]:
        entry: dict[str, Any] = {"path": _require("FalEmbeddingConfig", "path", path)}
        token_list = [part.strip() for part in (tokens or "").split(",") if part.strip()]
        if token_list:
            entry = {**entry, "tokens": token_list}
        return (_append_entry("FalEmbeddingConfig", chain, entry),)


class FalControlNetConfig:
    """Append one ControlNet conditioning entry to a JSON array."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build a `controlnets` JSON array entry ({path, control_image_url, "
        "conditioning_scale, start/end_percentage}) for fal-ai/flux-general and "
        "its variants (image-to-image, inpainting, differential-diffusion). "
        "Connect an IMAGE (auto-uploaded) or paste a control image URL."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "URL or Hugging Face path to the ControlNet weights. Feeds the "
                            "`controlnets` field of fal-ai/flux-general, "
                            "fal-ai/flux-general/image-to-image, fal-ai/flux-general/inpainting."
                        ),
                    },
                ),
                "conditioning_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Strength of the ControlNet guidance (ControlNet.conditioning_scale).",
                    },
                ),
                "start_percentage": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of total timesteps at which the ControlNet starts applying (0-1).",
                    },
                ),
                "end_percentage": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of total timesteps at which the ControlNet stops applying (0-1).",
                    },
                ),
            },
            "optional": {
                "control_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Control image (canny/depth/pose map, ...). Uploaded to fal "
                            "storage and sent as `control_image_url`. Overrides the URL widget."
                        ),
                    },
                ),
                "control_image_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Direct URL for the control image; used when no IMAGE is connected. "
                            "The schema requires one of the two."
                        ),
                    },
                ),
                "variant": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional variant when `path` is a Hugging Face repo key. Leave empty otherwise.",
                    },
                ),
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(
        self,
        path: str,
        conditioning_scale: float,
        start_percentage: float,
        end_percentage: float,
        control_image: Any = None,
        control_image_url: str = "",
        variant: str = "",
        chain: str = "",
    ) -> tuple[str]:
        node = "FalControlNetConfig"
        entry: dict[str, Any] = {
            "path": _require(node, "path", path),
            "control_image_url": _resolve_image_url(
                node, "control_image_url", control_image, control_image_url, required=True
            ),
            "conditioning_scale": float(conditioning_scale),
            "start_percentage": float(start_percentage),
            "end_percentage": float(end_percentage),
        }
        if (variant or "").strip():
            entry = {**entry, "variant": variant.strip()}
        return (_append_entry(node, chain, entry),)


class FalIPAdapterConfig:
    """Append one IP-Adapter entry to a JSON array."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build an `ip_adapters` JSON array entry ({path, image_encoder_path, "
        "image_url, scale}) for fal-ai/flux-general and its variants. Connect "
        "an IMAGE (auto-uploaded) or paste a reference image URL. For the older "
        "fal-ai/lora `ip_adapter` field (different keys) use FalKeyValue."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Hugging Face path to the IP-Adapter weights. Feeds the "
                            "`ip_adapters` field of fal-ai/flux-general, "
                            "fal-ai/flux-general/image-to-image, fal-ai/flux-general/rf-inversion."
                        ),
                    },
                ),
                "image_encoder_path": (
                    "STRING",
                    {
                        "default": "openai/clip-vit-large-patch14",
                        "tooltip": "Path to the image encoder for the IP-Adapter (IPAdapter.image_encoder_path).",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Strength of the IP-Adapter conditioning (IPAdapter.scale).",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference image for the IP-Adapter conditioning. Uploaded to fal "
                            "storage and sent as `image_url`. Overrides the URL widget."
                        ),
                    },
                ),
                "image_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Direct URL for the reference image; used when no IMAGE is connected. "
                            "The schema requires one of the two."
                        ),
                    },
                ),
                "weight_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional safetensors file name containing the IP-Adapter weights "
                            "(IPAdapter.weight_name). Leave empty otherwise."
                        ),
                    },
                ),
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(
        self,
        path: str,
        image_encoder_path: str,
        scale: float,
        image: Any = None,
        image_url: str = "",
        weight_name: str = "",
        chain: str = "",
    ) -> tuple[str]:
        node = "FalIPAdapterConfig"
        entry: dict[str, Any] = {
            "path": _require(node, "path", path),
            "image_encoder_path": _require(node, "image_encoder_path", image_encoder_path),
            "image_url": _resolve_image_url(node, "image_url", image, image_url, required=True),
            "scale": float(scale),
        }
        if (weight_name or "").strip():
            entry = {**entry, "weight_name": weight_name.strip()}
        return (_append_entry(node, chain, entry),)


class FalReferenceImage:
    """Append one Kling ElementInput (reference character/object) to a JSON array."""

    _MAX_REFERENCE_IMAGES = 3

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build an `elements` JSON array entry ({frontal_image_url, "
        "reference_image_urls}) for Kling Omni image endpoints "
        "(fal-ai/kling-image/o1, fal-ai/kling-image/o3/text-to-image, "
        "fal-ai/kling-image/o3/image-to-image). Images are auto-uploaded. "
        "Chain one builder per character/object element."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "frontal_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Frontal view of the character/object. Uploaded to fal storage and "
                            "sent as `frontal_image_url` inside the `elements` field of "
                            "fal-ai/kling-image/o1 and fal-ai/kling-image/o3 endpoints."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional batch of up to 3 additional views from different angles "
                            "(sent as `reference_image_urls`)."
                        ),
                    },
                ),
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(self, frontal_image: Any, reference_images: Any = None, chain: str = "") -> tuple[str]:
        node = "FalReferenceImage"
        entry: dict[str, Any] = {"frontal_image_url": ImageUtils.upload_image(frontal_image)}
        if reference_images is not None:
            urls = ImageUtils.prepare_images(reference_images)
            if len(urls) > self._MAX_REFERENCE_IMAGES:
                raise FalApiError(
                    node,
                    f"'reference_images' supports at most {self._MAX_REFERENCE_IMAGES} "
                    f"images per element (got {len(urls)})",
                )
            if urls:
                entry = {**entry, "reference_image_urls": urls}
        return (_append_entry(node, chain, entry),)


class FalMultiPromptShot:
    """Append one Kling multi-prompt shot ({prompt, duration}) to a JSON array."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Build a `multi_prompt` JSON array entry ({prompt, duration}) for Kling "
        "O3 video endpoints (fal-ai/kling-video/o3/standard/image-to-video, "
        "fal-ai/kling-video/o3/pro/text-to-video, .../4k variants). Chain one "
        "builder per shot to script a multi-shot video."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "The prompt for this shot. Feeds the `multi_prompt` field of "
                            "fal-ai/kling-video/o3 image-to-video / text-to-video / "
                            "reference-to-video endpoints."
                        ),
                    },
                ),
                "duration": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 15,
                        "tooltip": "Duration of this shot in seconds (1-15, sent as a string per the schema).",
                    },
                ),
            },
            "optional": {
                "chain": ("STRING", {"forceInput": True, "tooltip": _CHAIN_TOOLTIP}),
            },
        }

    def build(self, prompt: str, duration: int, chain: str = "") -> tuple[str]:
        node = "FalMultiPromptShot"
        entry = {
            "prompt": _require(node, "prompt", prompt),
            "duration": str(int(duration)),
        }
        return (_append_entry(node, chain, entry),)


def _typed_value(node: str, value: str, value_type: str) -> Any:
    """Coerce the FalKeyValue string widget into the selected JSON type."""
    if value_type == "string":
        return value
    text = value.strip()
    if value_type == "number":
        try:
            number = float(text)
        except ValueError as err:
            raise FalApiError(node, f"'value' is not a number: {text!r}") from err
        if not math.isfinite(number):
            raise FalApiError(node, f"'value' must be a finite number, got: {text!r}")
        return int(number) if number.is_integer() else number
    if value_type == "boolean":
        lowered = text.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise FalApiError(node, f"'value' is not a boolean (use true/false): {text!r}")
    # value_type == "json": nested arrays/objects/null, e.g. from another builder
    try:
        return json.loads(text)
    except ValueError as err:
        raise FalApiError(node, f"'value' is not valid JSON: {err}") from err


class FalKeyValue:
    """Merge one typed key/value pair into a JSON object (chainable)."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Generic escape hatch: build a JSON OBJECT one typed key at a time. "
        "Chain several to fill object fields like `audio_setting` / "
        "`voice_setting` (fal-ai/minimax-music/v2, fal-ai/minimax/speech-02-hd) "
        "or `validation` (fal-ai/ltx23-trainer-v2). Set value_type to `json` to "
        "nest arrays/objects, including outputs of the array builders."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Object key to set, e.g. sample_rate for `audio_setting` on "
                            "fal-ai/minimax-music/v2 or speed for `voice_setting` on "
                            "fal-ai/minimax/speech-02-hd."
                        ),
                    },
                ),
                "value": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value for the key, interpreted according to value_type.",
                    },
                ),
                "value_type": (
                    ["string", "number", "boolean", "json"],
                    {
                        "default": "string",
                        "tooltip": (
                            "How to encode the value: string as-is, number/boolean parsed, "
                            "json for nested objects/arrays (e.g. a builder output)."
                        ),
                    },
                ),
            },
            "optional": {
                "chain": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional: wire another FalKeyValue json output here to merge this "
                            "key into that object (later keys win)."
                        ),
                    },
                ),
            },
        }

    def build(self, key: str, value: str, value_type: str, chain: str = "") -> tuple[str]:
        node = "FalKeyValue"
        prior = _parse_chain(node, chain, dict)
        merged = {**prior, _require(node, "key", key): _typed_value(node, value, value_type)}
        return (json.dumps(merged),)


class FalJSONMerge:
    """Merge two builder outputs: arrays concatenate, objects merge (b wins)."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "merge"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Merge two JSON strings: two arrays concatenate (a then b), two objects "
        "merge with b overriding a. Useful to combine separately built chains "
        "before wiring them into one json field (e.g. two `loras` chains, or "
        "FalKeyValue objects for `audio_setting` / `validation`)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "a": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "First JSON array or object (a builder json output). Empty is allowed.",
                    },
                ),
                "b": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Second JSON array or object. Must be the same container type as "
                            "'a'; object keys in 'b' override 'a'."
                        ),
                    },
                ),
            },
        }

    @staticmethod
    def _parse(side: str, text: str) -> Any:
        stripped = (text or "").strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except ValueError as err:
            raise FalApiError("FalJSONMerge", f"'{side}' is not valid JSON: {err}") from err
        if not isinstance(parsed, (list, dict)):
            raise FalApiError(
                "FalJSONMerge",
                f"'{side}' must be a JSON array or object, got {type(parsed).__name__}",
            )
        return parsed

    def merge(self, a: str, b: str) -> tuple[str]:
        parsed_a = self._parse("a", a)
        parsed_b = self._parse("b", b)
        if parsed_a is None and parsed_b is None:
            raise FalApiError("FalJSONMerge", "Both 'a' and 'b' are empty; nothing to merge")
        if parsed_a is None or parsed_b is None:
            return (json.dumps(parsed_b if parsed_a is None else parsed_a),)
        if isinstance(parsed_a, list) and isinstance(parsed_b, list):
            return (json.dumps([*parsed_a, *parsed_b]),)
        if isinstance(parsed_a, dict) and isinstance(parsed_b, dict):
            return (json.dumps({**parsed_a, **parsed_b}),)
        raise FalApiError(
            "FalJSONMerge",
            "'a' and 'b' must both be arrays or both be objects "
            f"(got {type(parsed_a).__name__} and {type(parsed_b).__name__})",
        )


NODE_CLASS_MAPPINGS = {
    "FalLoRAConfig_fal": FalLoRAConfig,
    "FalEmbeddingConfig_fal": FalEmbeddingConfig,
    "FalControlNetConfig_fal": FalControlNetConfig,
    "FalIPAdapterConfig_fal": FalIPAdapterConfig,
    "FalReferenceImage_fal": FalReferenceImage,
    "FalMultiPromptShot_fal": FalMultiPromptShot,
    "FalKeyValue_fal": FalKeyValue,
    "FalJSONMerge_fal": FalJSONMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalLoRAConfig_fal": "LoRA Config (fal)",
    "FalEmbeddingConfig_fal": "Embedding Config (fal)",
    "FalControlNetConfig_fal": "ControlNet Config (fal)",
    "FalIPAdapterConfig_fal": "IP-Adapter Config (fal)",
    "FalReferenceImage_fal": "Reference Image Element (fal)",
    "FalMultiPromptShot_fal": "Multi-Prompt Shot (fal)",
    "FalKeyValue_fal": "Key/Value JSON (fal)",
    "FalJSONMerge_fal": "JSON Merge (fal)",
}
