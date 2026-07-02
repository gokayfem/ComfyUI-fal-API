"""Image tensor helpers and API result processing for ComfyUI-fal-API.

ComfyUI IMAGE convention: float32 tensors in [0, 1] with shape (B, H, W, C).
"""

from __future__ import annotations

import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from .config import FalConfig
from .errors import FalApiError, raise_fal_error
from .logger import logger

_DOWNLOAD_TIMEOUT = (10, 180)
_MAX_PARALLEL_TRANSFERS = 8


def _safe_unlink(path: str) -> None:
    """Delete a temp file, ignoring errors."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _download_image_array(url: str) -> np.ndarray:
    """Download an image URL and return a float32 (H, W, 3) array in [0, 1]."""
    response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def _download_image_arrays(urls: list[str]) -> list[np.ndarray]:
    """Download image URLs (in parallel when multiple), preserving order."""
    if len(urls) == 1:
        return [_download_image_array(urls[0])]
    max_workers = min(len(urls), _MAX_PARALLEL_TRANSFERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_download_image_array, urls))


def _split_image_batch(images: Any) -> list[Any]:
    """Split an IMAGE input into a list of single images, preserving order."""
    if isinstance(images, torch.Tensor):
        if images.ndim == 4 and images.shape[0] > 1:
            return [images[i : i + 1] for i in range(images.shape[0])]
        return [images]
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


class ImageUtils:
    """Utility functions for image processing and uploads."""

    @staticmethod
    def tensor_to_pil(image: Any) -> Image.Image:
        """Convert an image tensor (or array-like) to a PIL Image."""
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().numpy()
            else:
                image_np = np.array(image)

            if image_np.ndim == 4:
                image_np = image_np[0]  # Drop batch dimension
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Grayscale -> RGB
            elif (
                image_np.ndim == 3
                and image_np.shape[0] == 3
                and image_np.shape[2] not in (1, 3, 4)
            ):
                image_np = np.transpose(image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)

            if image_np.dtype in (np.float32, np.float64):
                image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

            return Image.fromarray(image_np)
        except Exception as exc:
            logger.error("Failed to convert tensor to PIL image: %s", exc)
            raise FalApiError(
                "image-utils", f"Failed to convert tensor to image: {exc}"
            ) from exc

    @staticmethod
    def upload_image(image: Any) -> str:
        """Upload an image tensor to fal.ai and return its URL."""
        pil_image = ImageUtils.tensor_to_pil(image)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                pil_image.save(temp_file, format="PNG")
            return ImageUtils.upload_file(temp_path)
        finally:
            if temp_path is not None:
                _safe_unlink(temp_path)

    @staticmethod
    def upload_file(file_path: Any) -> str:
        """Upload a local file to fal.ai and return its URL."""
        try:
            client = FalConfig().get_client()
            return client.upload_file(file_path)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("Failed to upload file %s: %s", file_path, exc)
            raise_fal_error("file-upload", exc)

    @staticmethod
    def mask_to_image(mask: torch.Tensor) -> torch.Tensor:
        """Convert a MASK tensor to an IMAGE tensor (B, H, W, 3)."""
        return (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )

    @staticmethod
    def prepare_images(images: Any) -> list[str]:
        """Upload image input(s) to fal.ai in parallel, preserving order."""
        if images is None:
            return []
        singles = _split_image_batch(images)
        if not singles:
            return []
        if len(singles) == 1:
            return [ImageUtils.upload_image(singles[0])]
        max_workers = min(len(singles), _MAX_PARALLEL_TRANSFERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(ImageUtils.upload_image, singles))


class ResultProcessor:
    """Utility functions for turning API results into ComfyUI tensors."""

    @staticmethod
    def process_image_result(result: dict[str, Any]) -> tuple:
        """Process a multi-image result ({"images": [{"url": ...}, ...]})."""
        try:
            urls = [img_info["url"] for img_info in result["images"]]
            if not urls:
                raise ValueError("API result contained no images")
            arrays = _download_image_arrays(urls)
            stacked = np.stack(arrays, axis=0)
            return (torch.from_numpy(stacked),)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("Failed to process image result: %s", exc)
            raise FalApiError(
                "image-result", f"Failed to process image result: {exc}"
            ) from exc

    @staticmethod
    def process_single_image_result(result: dict[str, Any]) -> tuple:
        """Process a single-image result ({"image": {"url": ...}})."""
        try:
            img_array = _download_image_array(result["image"]["url"])
            stacked = np.stack([img_array], axis=0)
            return (torch.from_numpy(stacked),)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("Failed to process single image result: %s", exc)
            raise FalApiError(
                "image-result", f"Failed to process single image result: {exc}"
            ) from exc

    @staticmethod
    def create_blank_image() -> tuple:
        """Create a blank black 512x512 IMAGE tensor (kept for compatibility)."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)
