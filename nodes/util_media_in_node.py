"""Utility loader nodes: bring images/audio/folders into ComfyUI from URLs and disk.

ComfyUI IMAGE convention: float32 tensors in [0, 1] with shape (B, H, W, C).
"""

from __future__ import annotations

import glob
import io
import os
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from .fal_utils import FalApiError, MediaUtils, logger

_CATEGORY = "FAL/Utils/Load"
_DOWNLOAD_TIMEOUT = (10, 180)
_DEFAULT_FOLDER_PATTERN = "*.png,*.jpg,*.jpeg,*.webp"


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated string into stripped, non-empty parts."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _validate_http_url(node_name: str, url: str) -> str:
    """Validate that a URL is a non-empty http(s) URL and return it stripped."""
    stripped = (url or "").strip()
    if not stripped:
        raise FalApiError(
            node_name, "'url' is empty. Provide an http(s) URL to a media file."
        )
    if not stripped.startswith(("http://", "https://")):
        raise FalApiError(
            node_name,
            f"Invalid URL '{stripped}'. Only http(s) URLs are supported.",
        )
    return stripped


def _download_pil_image(node_name: str, url: str) -> Image.Image:
    """Download a URL and decode it as an RGB PIL image."""
    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except FalApiError:
        raise
    except Exception as exc:
        logger.error("%s: failed to download image %s: %s", node_name, url, exc)
        raise FalApiError(
            node_name,
            f"Failed to download or decode image from '{url}': {exc}",
        ) from exc


def _images_to_batch_tensor(
    node_name: str, images: list[Image.Image], labels: list[str]
) -> torch.Tensor:
    """Stack RGB PIL images into a float32 (B, H, W, C) tensor in [0, 1].

    Images whose size differs from the first image are resized to match
    (with a warning) so the batch stays valid.
    """
    first_size = images[0].size  # (W, H)
    arrays: list[np.ndarray] = []
    for img, label in zip(images, labels):
        if img.size != first_size:
            logger.warning(
                "%s: '%s' is %sx%s; resizing to %sx%s to match the first image",
                node_name,
                label,
                img.size[0],
                img.size[1],
                first_size[0],
                first_size[1],
            )
            img = img.resize(first_size, Image.LANCZOS)
        arrays.append(np.array(img).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0))


class FalLoadImageURL:
    """Load one or more images from http(s) URLs into an IMAGE batch."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Download an image from an http(s) URL into a ComfyUI IMAGE tensor. "
        "Accepts a comma-separated list of URLs to build a batch; images with "
        "differing sizes are resized to match the first."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "http(s) URL of the image to load. A comma-separated "
                            "list of URLs produces a batched IMAGE; mismatched "
                            "sizes are resized to the first image's size. URLs "
                            "containing literal commas are not supported in list mode."
                        ),
                    },
                ),
            },
        }

    def load(self, url: str) -> tuple[torch.Tensor]:
        node_name = "FalLoadImageURL"
        urls = [_validate_http_url(node_name, part) for part in _split_csv(url)]
        if not urls:
            raise FalApiError(
                node_name,
                "'url' is empty. Provide an http(s) URL (or a comma-separated "
                "list of URLs) to image file(s).",
            )
        images = [_download_pil_image(node_name, u) for u in urls]
        return (_images_to_batch_tensor(node_name, images, urls),)


class FalLoadAudioURL:
    """Load audio from an http(s) URL into a ComfyUI AUDIO output."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Download and decode an audio file from an http(s) URL into a native "
        "ComfyUI AUDIO output (waveform + sample rate)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "http(s) URL of the audio file to download and "
                            "decode (e.g. the audio_url output of a fal node)."
                        ),
                    },
                ),
            },
        }

    def load(self, url: str) -> tuple[dict[str, Any]]:
        node_name = "FalLoadAudioURL"
        validated = _validate_http_url(node_name, url)
        try:
            return (MediaUtils.audio_from_url(validated),)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("%s: failed to load audio %s: %s", node_name, validated, exc)
            raise FalApiError(
                node_name,
                f"Failed to load audio from '{validated}': {exc}",
            ) from exc


def _resolve_folder(node_name: str, folder_path: str) -> str:
    """Expand and validate a folder path, returning its absolute form."""
    expanded = os.path.expanduser((folder_path or "").strip())
    if not expanded:
        raise FalApiError(
            node_name, "'folder_path' is empty. Provide a path to a folder."
        )
    if not os.path.isdir(expanded):
        raise FalApiError(
            node_name,
            f"Folder not found: '{expanded}'. Provide an existing folder path.",
        )
    return os.path.abspath(expanded)


def _glob_folder_files(folder: str, patterns: list[str]) -> list[str]:
    """Glob a folder with each pattern, deduplicated, unordered."""
    matched: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(folder, pattern)):
            if os.path.isfile(path):
                matched.add(os.path.abspath(path))
    return list(matched)


def _sort_files(files: list[str], sort: str) -> list[str]:
    """Sort file paths deterministically by name or modification time."""
    if sort == "modified":
        return sorted(files, key=lambda path: (os.path.getmtime(path), path))
    return sorted(files)


class FalLoadImageFolder:
    """Load a folder of images from disk into a single IMAGE batch."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Load every image matching the pattern(s) in a local folder into one "
        "IMAGE batch. Mixed sizes are resized to the first image's dimensions."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to a local folder of images. '~' expands to "
                            "your home directory."
                        ),
                    },
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": _DEFAULT_FOLDER_PATTERN,
                        "tooltip": (
                            "Comma-separated glob pattern(s) selecting which "
                            "files to load, e.g. '*.png,*.jpg'."
                        ),
                    },
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Maximum number of images to load from the folder.",
                    },
                ),
                "sort": (
                    ["name", "modified"],
                    {
                        "default": "name",
                        "tooltip": (
                            "Order in which files are loaded: alphabetical by "
                            "'name' or oldest-first by 'modified' time."
                        ),
                    },
                ),
            },
        }

    def load(
        self, folder_path: str, pattern: str, max_images: int, sort: str
    ) -> tuple[torch.Tensor, int]:
        node_name = "FalLoadImageFolder"
        folder = _resolve_folder(node_name, folder_path)
        patterns = _split_csv(pattern) or _split_csv(_DEFAULT_FOLDER_PATTERN)
        files = _sort_files(_glob_folder_files(folder, patterns), sort)[:max_images]
        if not files:
            raise FalApiError(
                node_name,
                f"No files matching '{', '.join(patterns)}' found in '{folder}'. "
                "Adjust 'pattern' or point 'folder_path' at a folder with images.",
            )
        images = [self._open_image(node_name, path) for path in files]
        batch = _images_to_batch_tensor(node_name, images, files)
        return (batch, len(files))

    @staticmethod
    def _open_image(node_name: str, path: str) -> Image.Image:
        """Open a local image file as RGB, normalizing failures."""
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as exc:
            logger.error("%s: failed to open image %s: %s", node_name, path, exc)
            raise FalApiError(
                node_name,
                f"Failed to open image '{path}': {exc}. Remove or exclude the "
                "file via 'pattern' and retry.",
            ) from exc


def _normalize_extensions(extensions: str) -> list[str] | None:
    """Parse a comma-separated extension filter; empty means no filter."""
    parts = [part.lstrip("*").lower() for part in _split_csv(extensions)]
    normalized = [part if part.startswith(".") else f".{part}" for part in parts]
    cleaned = [part for part in normalized if part != "."]
    return cleaned or None


class FalUploadFolderAsZip:
    """Zip a local folder and upload the archive to fal.ai, returning its URL."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_url",)
    FUNCTION = "upload"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Zip a local folder (optionally recursive / filtered by extension) and "
        "upload the archive to fal.ai storage, returning the ZIP's URL — handy "
        "for endpoints that take a training-data archive."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to the local folder to zip and upload. '~' "
                            "expands to your home directory."
                        ),
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include files from subfolders in the ZIP.",
                    },
                ),
                "extensions": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Comma-separated file extensions to include, e.g. "
                            "'.png,.jpg'. Leave empty to include all files."
                        ),
                    },
                ),
            },
        }

    def upload(
        self, folder_path: str, recursive: bool, extensions: str
    ) -> tuple[str]:
        node_name = "FalUploadFolderAsZip"
        folder = _resolve_folder(node_name, folder_path)
        archive_utils = self._load_archive_utils(node_name)
        include_extensions = _normalize_extensions(extensions)
        try:
            zip_path = archive_utils.zip_folder(
                folder, include_extensions=include_extensions, recursive=recursive
            )
            return (archive_utils.upload_zip(zip_path),)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("%s: failed to zip/upload %s: %s", node_name, folder, exc)
            raise FalApiError(
                node_name,
                f"Failed to zip and upload folder '{folder}': {exc}",
            ) from exc

    @staticmethod
    def _load_archive_utils(node_name: str) -> Any:
        """Lazily import ArchiveUtils, degrading with an actionable error."""
        try:
            from .fal_utils import ArchiveUtils

            return ArchiveUtils
        except ImportError as exc:
            raise FalApiError(
                node_name,
                "Archive utilities are unavailable in this install "
                f"({exc}). Update/reinstall ComfyUI-fal-API so that "
                "nodes/utils/archive.py is present.",
            ) from exc


NODE_CLASS_MAPPINGS = {
    "FalLoadImageURL_fal": FalLoadImageURL,
    "FalLoadAudioURL_fal": FalLoadAudioURL,
    "FalLoadImageFolder_fal": FalLoadImageFolder,
    "FalUploadFolderAsZip_fal": FalUploadFolderAsZip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalLoadImageURL_fal": "Load Image from URL (fal)",
    "FalLoadAudioURL_fal": "Load Audio from URL (fal)",
    "FalLoadImageFolder_fal": "Load Image Folder (fal)",
    "FalUploadFolderAsZip_fal": "Upload Folder as ZIP URL (fal)",
}
