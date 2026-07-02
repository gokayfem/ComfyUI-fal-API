"""Zip archive helpers for dataset preparation (LoRA training uploads)."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from typing import Any

import torch

from .config import FalConfig
from .errors import FalApiError
from .images import ImageUtils
from .logger import logger

_MODEL_NAME = "archive"


def _safe_unlink(path: str | None) -> None:
    """Delete a temp file, ignoring errors."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _split_frames(images: Any) -> list[Any]:
    """Split an IMAGE input (batch tensor, list, or single image) into frames."""
    if images is None:
        return []
    if isinstance(images, torch.Tensor):
        if images.ndim == 4:
            return [images[i] for i in range(images.shape[0])]
        return [images]
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def _normalize_extensions(extensions: Any) -> list[str] | None:
    """Normalize an extension filter to lowercase dot-prefixed suffixes."""
    if not extensions:
        return None
    normalized = []
    for ext in extensions:
        cleaned = str(ext).strip().lower()
        if not cleaned:
            continue
        normalized.append(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return normalized or None


def _matches_filter(file_name: str, extensions: list[str] | None) -> bool:
    """Whether a file passes the hidden-file and extension filters."""
    if file_name.startswith("."):
        return False
    if extensions is None:
        return True
    return os.path.splitext(file_name)[1].lower() in extensions


def _collect_folder_files(
    folder: str, extensions: list[str] | None, recursive: bool
) -> list[str]:
    """List matching files in a folder (sorted, hidden entries skipped)."""
    if not recursive:
        return [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name))
            and _matches_filter(name, extensions)
        ]
    matches: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if _matches_filter(name, extensions):
                matches.append(os.path.join(root, name))
    return matches


def _new_temp_zip_path() -> str:
    """Reserve a temp .zip path and return it."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        return temp_zip.name


class ArchiveUtils:
    """Utility functions for building and uploading zip archives."""

    @staticmethod
    def zip_images(
        images: Any,
        captions: list[str] | None = None,
        name_prefix: str = "image",
    ) -> str:
        """Zip an IMAGE batch as image_0.png, image_1.png, ... and return the local zip path.

        ``captions`` (optional, one per image, entries may be "") also writes
        image_0.txt, image_1.txt, ... — the standard LoRA-training caption
        layout. The caller is responsible for uploading/deleting the zip
        (see ``upload_zip``).
        """
        frames = _split_frames(images)
        if not frames:
            raise FalApiError(
                _MODEL_NAME,
                "No images provided to zip. Connect an IMAGE batch with at least one frame.",
            )
        if captions is not None and len(captions) != len(frames):
            raise FalApiError(
                _MODEL_NAME,
                f"Caption count ({len(captions)}) does not match image count ({len(frames)}). "
                "Provide exactly one caption per image (blank entries are allowed) or none at all.",
            )
        prefix = (name_prefix or "image").strip() or "image"

        zip_path: str | None = None
        try:
            zip_path = _new_temp_zip_path()
            with zipfile.ZipFile(zip_path, "w") as zip_file:
                for index, frame in enumerate(frames):
                    pil_image = ImageUtils.tensor_to_pil(frame)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    zip_file.writestr(f"{prefix}_{index}.png", buffer.getvalue())
                    if captions is not None:
                        zip_file.writestr(f"{prefix}_{index}.txt", captions[index])
            return zip_path
        except FalApiError:
            _safe_unlink(zip_path)
            raise
        except Exception as exc:
            _safe_unlink(zip_path)
            logger.error("Failed to create image zip: %s", exc)
            raise FalApiError(
                _MODEL_NAME, f"Failed to create image zip: {exc}"
            ) from exc

    @staticmethod
    def zip_folder(
        folder_path: str,
        include_extensions: list[str] | None = None,
        recursive: bool = False,
    ) -> str:
        """Zip a folder's files and return the local zip path.

        ``include_extensions`` filters by suffix (e.g. [".png", ".txt"]); None
        includes everything. Hidden files/directories are always skipped.
        Non-recursive by default; arcnames are relative to the folder.
        """
        if not folder_path or not isinstance(folder_path, str) or not folder_path.strip():
            raise FalApiError(
                _MODEL_NAME,
                "folder_path is empty. Provide the path to a folder of dataset files.",
            )
        folder = os.path.abspath(os.path.expanduser(folder_path.strip()))
        if not os.path.isdir(folder):
            raise FalApiError(
                _MODEL_NAME,
                f"Folder not found: {folder}. Provide the path to an existing directory.",
            )

        extensions = _normalize_extensions(include_extensions)
        files = _collect_folder_files(folder, extensions, recursive)
        if not files:
            suffix_hint = f" matching extensions {extensions}" if extensions else ""
            raise FalApiError(
                _MODEL_NAME,
                f"No files{suffix_hint} found in {folder}. "
                "Check the folder contents, the extension filter, and the recursive flag.",
            )

        # Folder zips get uploaded to fal's CDN: log loudly what is being read
        # and cap runaway/hostile selections ([archive] section in config.ini).
        total_bytes = sum(os.path.getsize(f) for f in files)
        max_files = int(FalConfig().get_setting("archive", "max_files", 5000))
        max_mb = float(FalConfig().get_setting("archive", "max_total_mb", 2048))
        if len(files) > max_files or total_bytes > max_mb * 1024 * 1024:
            raise FalApiError(
                _MODEL_NAME,
                f"Refusing to zip {len(files)} file(s) / {total_bytes / 1048576:.1f} MiB "
                f"from {folder} — over the [archive] limits (max_files={max_files}, "
                f"max_total_mb={max_mb:g}). Narrow the folder/extensions or raise the "
                "limits in config.ini.",
            )
        logger.info(
            "archive: zipping %d file(s) (%.1f MiB) from %s",
            len(files),
            total_bytes / 1048576,
            folder,
        )

        zip_path: str | None = None
        try:
            zip_path = _new_temp_zip_path()
            with zipfile.ZipFile(zip_path, "w") as zip_file:
                for file_path in files:
                    zip_file.write(file_path, os.path.relpath(file_path, folder))
            return zip_path
        except Exception as exc:
            _safe_unlink(zip_path)
            logger.error("Failed to zip folder %s: %s", folder, exc)
            raise FalApiError(
                _MODEL_NAME, f"Failed to zip folder {folder}: {exc}"
            ) from exc

    @staticmethod
    def upload_zip(zip_path: str) -> str:
        """Upload a local zip to fal.ai and return its URL; the zip is always deleted."""
        if not zip_path or not os.path.isfile(zip_path):
            raise FalApiError(
                _MODEL_NAME,
                f"Zip file not found: {zip_path}. Build it with zip_images/zip_folder first.",
            )
        try:
            return ImageUtils.upload_file(zip_path)
        finally:
            _safe_unlink(zip_path)
