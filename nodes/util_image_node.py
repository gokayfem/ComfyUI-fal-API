"""Image utility nodes: labeled grids, preset resizing, and base64 conversion."""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .fal_utils import FalApiError, ImageUtils, logger

_CATEGORY = "FAL/Utils/Image"

# fal image_size preset -> (width, height)
_PRESET_SIZES = {
    "square_hd": (1024, 1024),
    "square": (512, 512),
    "portrait_4_3": (768, 1024),
    "portrait_16_9": (576, 1024),
    "landscape_4_3": (1024, 768),
    "landscape_16_9": (1024, 576),
}
_CUSTOM_PRESET = "custom"

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

_GRID_BG = (24, 24, 24)
_LABEL_BG = (16, 16, 16)
_LABEL_FG = (235, 235, 235)
_LABEL_MARGIN = 4
_ELLIPSIS = "..."

_B64_WHITESPACE = re.compile(r"\s+")
_MIME_BY_FORMAT = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


def _pils_to_tensor(pils: list[Image.Image]) -> torch.Tensor:
    """Stack same-sized PIL images into a float32 (B, H, W, 3) IMAGE tensor."""
    arrays = [np.array(pil.convert("RGB")).astype(np.float32) / 255.0 for pil in pils]
    return torch.from_numpy(np.stack(arrays, axis=0))


def _image_input_to_pils(images: Any) -> list[Image.Image]:
    """Convert an IMAGE input (batch tensor or list of tensors) to PIL images."""
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        items: list[Any] = [images[i] for i in range(images.shape[0])]
    elif isinstance(images, (list, tuple)):
        items = list(images)
    else:
        items = [images]
    if not items:
        raise FalApiError("FalImageGrid", "IMAGE input contained no images")
    return [ImageUtils.tensor_to_pil(item) for item in items]


def _letterbox(pil: Image.Image, width: int, height: int, fill: tuple[int, int, int]) -> Image.Image:
    """Fit an image inside (width, height) preserving aspect, padded with fill."""
    scale = min(width / pil.width, height / pil.height)
    new_size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
    resized = pil.convert("RGB").resize(new_size, _LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    offset = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def _truncate_label(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> str:
    """Truncate text with an ellipsis so it fits within max_width pixels."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = text[:end].rstrip() + _ELLIPSIS
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    return _ELLIPSIS


def _draw_label(
    canvas: Image.Image, text: str, x: int, y: int, cell_width: int, label_height: int
) -> None:
    """Draw one centered label line on its dark strip below a cell."""
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + cell_width - 1, y + label_height - 1), fill=_LABEL_BG)
    if not text:
        return
    font = ImageFont.load_default()
    fitted = _truncate_label(draw, text, font, cell_width - 2 * _LABEL_MARGIN)
    text_width = draw.textlength(fitted, font=font)
    bbox = font.getbbox(fitted)
    text_height = bbox[3] - bbox[1]
    text_x = x + max(_LABEL_MARGIN, (cell_width - text_width) // 2)
    text_y = y + max(0, (label_height - text_height) // 2) - bbox[1]
    draw.text((text_x, text_y), fitted, font=font, fill=_LABEL_FG)


def _grid_shape(count: int, columns: int) -> tuple[int, int]:
    """Resolve (columns, rows) for a grid; columns == 0 means auto square-ish."""
    cols = columns if columns > 0 else math.ceil(math.sqrt(count))
    cols = max(1, min(cols, count))
    return cols, math.ceil(count / cols)


def _compose_grid(
    pils: list[Image.Image], labels: list[str], columns: int, padding: int, label_height: int
) -> Image.Image:
    """Lay out letterboxed cells (plus optional label strips) on a dark canvas."""
    cell_w = max(pil.width for pil in pils)
    cell_h = max(pil.height for pil in pils)
    strip_h = label_height if labels else 0
    cols, rows = _grid_shape(len(pils), columns)
    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * (cell_h + strip_h) + (rows + 1) * padding
    canvas = Image.new("RGB", (total_w, total_h), _GRID_BG)
    for i, pil in enumerate(pils):
        col, row = i % cols, i // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + strip_h + padding)
        canvas.paste(_letterbox(pil, cell_w, cell_h, _GRID_BG), (x, y))
        if strip_h:
            text = labels[i] if i < len(labels) else ""
            _draw_label(canvas, text, x, y + cell_h, cell_w, strip_h)
    return canvas


class FalImageGrid:
    """Compose an image batch into a single labeled contact-sheet grid."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compose"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Arrange a batch of images into one grid image with optional text "
        "labels under each cell. Mixed sizes are letterboxed into uniform "
        "cells on a dark background — handy for comparing seeds, prompts, "
        "or models side by side."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of images to arrange into a grid"}),
            },
            "optional": {
                "labels": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "One label per line, matched to images in batch order. "
                            "Leave empty for no label strips."
                        ),
                    },
                ),
                "columns": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Number of grid columns; 0 = auto (roughly square)",
                    },
                ),
                "cell_padding": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Pixels of dark padding around each cell",
                    },
                ),
                "label_height": (
                    "INT",
                    {
                        "default": 28,
                        "min": 12,
                        "max": 128,
                        "tooltip": "Height in pixels of the label strip under each cell",
                    },
                ),
            },
        }

    def compose(
        self,
        images: Any,
        labels: str = "",
        columns: int = 0,
        cell_padding: int = 8,
        label_height: int = 28,
    ) -> tuple[torch.Tensor]:
        pils = _image_input_to_pils(images)
        label_lines = [line.strip() for line in labels.splitlines()] if labels.strip() else []
        try:
            grid = _compose_grid(pils, label_lines, int(columns), int(cell_padding), int(label_height))
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("FalImageGrid: failed to compose grid: %s", exc)
            raise FalApiError("FalImageGrid", f"Failed to compose image grid: {exc}") from exc
        logger.debug("FalImageGrid: composed %d cells into %dx%d", len(pils), grid.width, grid.height)
        return (_pils_to_tensor([grid]),)


def _resize_one(pil: Image.Image, width: int, height: int, mode: str) -> Image.Image:
    """Resize a single PIL image to (width, height) using the given mode."""
    source = pil.convert("RGB")
    if mode == "stretch":
        return source.resize((width, height), _LANCZOS)
    if mode == "contain_pad":
        return _letterbox(source, width, height, (0, 0, 0))
    # cover_crop: scale to fully cover the target, then center-crop
    scale = max(width / source.width, height / source.height)
    scaled = source.resize(
        (max(width, round(source.width * scale)), max(height, round(source.height * scale))),
        _LANCZOS,
    )
    left = (scaled.width - width) // 2
    top = (scaled.height - height) // 2
    return scaled.crop((left, top, left + width, top + height))


class FalResizeToPreset:
    """Resize images to an exact fal image_size preset (or custom dimensions)."""

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Resize images to the exact pixel dimensions of a fal image_size "
        "preset (square_hd, portrait_16_9, ...) or custom width/height. "
        "Choose cover (crop), contain (letterbox), or stretch. Batch-safe."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image (or batch) to resize"}),
                "preset": (
                    [*_PRESET_SIZES.keys(), _CUSTOM_PRESET],
                    {
                        "default": "square_hd",
                        "tooltip": (
                            "fal image_size preset: square_hd=1024x1024, square=512x512, "
                            "portrait_4_3=768x1024, portrait_16_9=576x1024, "
                            "landscape_4_3=1024x768, landscape_16_9=1024x576. "
                            "'custom' uses the width/height inputs."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 14142,
                        "step": 8,
                        "tooltip": "Target width in pixels (used when preset is 'custom')",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 14142,
                        "step": 8,
                        "tooltip": "Target height in pixels (used when preset is 'custom')",
                    },
                ),
                "mode": (
                    ["cover_crop", "contain_pad", "stretch"],
                    {
                        "default": "cover_crop",
                        "tooltip": (
                            "cover_crop: fill the frame and center-crop the overflow; "
                            "contain_pad: fit inside and letterbox with black bars; "
                            "stretch: ignore aspect ratio"
                        ),
                    },
                ),
            },
        }

    def resize(
        self,
        image: Any,
        preset: str = "square_hd",
        width: int = 1024,
        height: int = 1024,
        mode: str = "cover_crop",
    ) -> tuple[torch.Tensor, int, int]:
        if preset == _CUSTOM_PRESET:
            target_w, target_h = int(width), int(height)
        elif preset in _PRESET_SIZES:
            target_w, target_h = _PRESET_SIZES[preset]
        else:
            raise FalApiError("FalResizeToPreset", f"Unknown preset: {preset!r}")
        if target_w < 1 or target_h < 1:
            raise FalApiError("FalResizeToPreset", f"Invalid target size: {target_w}x{target_h}")

        pils = _image_input_to_pils(image)
        try:
            resized = [_resize_one(pil, target_w, target_h, mode) for pil in pils]
        except Exception as exc:
            logger.error("FalResizeToPreset: resize failed: %s", exc)
            raise FalApiError("FalResizeToPreset", f"Failed to resize image: {exc}") from exc
        return (_pils_to_tensor(resized), target_w, target_h)


class FalImageToBase64:
    """Encode an image as a base64 string (optionally a data: URI)."""

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Encode the first image of a batch as base64 text, optionally "
        "wrapped in a data: URI — useful for APIs that accept inline "
        "base64 images instead of URLs."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to encode (first of the batch is used)"}),
                "format": (
                    ["png", "jpeg", "webp"],
                    {"default": "png", "tooltip": "Encoding format; png and webp are lossless-capable"},
                ),
                "data_uri": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prefix with 'data:image/...;base64,' (most APIs expect this)",
                    },
                ),
            },
        }

    def encode(self, image: Any, format: str = "png", data_uri: bool = True) -> tuple[str]:
        fmt = (format or "png").lower()
        if fmt not in _MIME_BY_FORMAT:
            raise FalApiError("FalImageToBase64", f"Unsupported format: {format!r}")
        pil = ImageUtils.tensor_to_pil(image).convert("RGB")
        try:
            buffer = io.BytesIO()
            save_kwargs = {"lossless": True} if fmt == "webp" else {}
            pil.save(buffer, format=fmt.upper(), **save_kwargs)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception as exc:
            logger.error("FalImageToBase64: encoding failed: %s", exc)
            raise FalApiError("FalImageToBase64", f"Failed to encode image as {fmt}: {exc}") from exc
        if data_uri:
            return (f"data:{_MIME_BY_FORMAT[fmt]};base64,{encoded}",)
        return (encoded,)


class FalBase64ToImage:
    """Decode a base64 string (raw or data: URI) into an IMAGE tensor."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Decode base64 image data — either a raw base64 string or a full "
        "'data:image/...;base64,...' URI — into a ComfyUI IMAGE."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "data": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Raw base64 image data, or a data:image/...;base64,... URI",
                    },
                ),
            },
        }

    def decode(self, data: str) -> tuple[torch.Tensor]:
        payload = (data or "").strip()
        if payload.startswith("data:"):
            _, _, payload = payload.partition(",")
        payload = _B64_WHITESPACE.sub("", payload)
        if not payload:
            raise FalApiError("FalBase64ToImage", "No base64 data provided")
        try:
            raw = base64.b64decode(payload)
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            logger.error("FalBase64ToImage: decoding failed: %s", exc)
            raise FalApiError("FalBase64ToImage", f"Failed to decode base64 image: {exc}") from exc
        return (_pils_to_tensor([pil]),)


NODE_CLASS_MAPPINGS = {
    "FalImageGrid_fal": FalImageGrid,
    "FalResizeToPreset_fal": FalResizeToPreset,
    "FalImageToBase64_fal": FalImageToBase64,
    "FalBase64ToImage_fal": FalBase64ToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalImageGrid_fal": "Image Grid with Labels (fal)",
    "FalResizeToPreset_fal": "Resize to fal Preset (fal)",
    "FalImageToBase64_fal": "Image → Base64 (fal)",
    "FalBase64ToImage_fal": "Base64 → Image (fal)",
}
