"""Dataset preparation utility nodes (zip building, frame extraction, captioning)."""

from __future__ import annotations

import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .fal_utils import (
    ApiHandler,
    ArchiveUtils,
    FalApiError,
    FalConfig,
    ImageUtils,
    MediaUtils,
    logger,
)

# Initialize FalConfig
fal_config = FalConfig()

_CATEGORY = "FAL/Utils/Dataset"
_ARCHIVE_MODEL = "archive"
_VISION_ENDPOINT = "openrouter/router/vision"
_MAX_CAPTION_WORKERS = 8
_STREAM_CHUNK_SIZE = 1 << 20  # 1 MiB


def _safe_unlink(path: str | None) -> None:
    """Delete a temp file, ignoring errors."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _split_caption_lines(captions: str) -> list[str] | None:
    """Split a multiline caption field into one caption per line (None if empty)."""
    if not captions or not captions.strip():
        return None
    return captions.splitlines()


def _video_to_local_path(video: Any) -> tuple[str, bool]:
    """Resolve a VIDEO input to a local file path. Returns (path, is_temp)."""
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else video
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return MediaUtils.download_url_to_temp(source, ".mp4"), True
        if not os.path.isfile(source):
            raise FalApiError(
                _ARCHIVE_MODEL,
                f"Video file not found: {source}. Connect a valid VIDEO input.",
            )
        return source, False
    if hasattr(source, "read"):
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = temp_file.name
                while True:
                    chunk = source.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    temp_file.write(chunk)
            return temp_path, True
        except Exception as exc:
            _safe_unlink(temp_path)
            raise FalApiError(
                _ARCHIVE_MODEL, f"Failed to buffer video stream to disk: {exc}"
            ) from exc
    raise FalApiError(
        _ARCHIVE_MODEL,
        "Unsupported VIDEO input: could not resolve a local file, URL, or stream from it.",
    )


def _extract_frames_to_zip(video_path: str, every_nth: int, max_frames: int) -> str:
    """Decode a video with cv2, sample every Nth frame as PNG into a zip, return the zip path."""
    try:
        import cv2
    except ImportError as exc:
        raise FalApiError(
            _ARCHIVE_MODEL,
            "opencv-python is required to extract video frames. "
            "Install it with 'pip install opencv-python'.",
        ) from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FalApiError(
            _ARCHIVE_MODEL,
            f"Could not open video for decoding: {video_path}. "
            "Check that the input is a valid video file.",
        )

    zip_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            zip_path = temp_zip.name
        saved = 0
        index = 0
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            while saved < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % every_nth == 0:
                    encoded, buffer = cv2.imencode(".png", frame)
                    if not encoded:
                        raise FalApiError(
                            _ARCHIVE_MODEL, f"Failed to encode frame {index} as PNG."
                        )
                    zip_file.writestr(f"frame_{saved:05d}.png", buffer.tobytes())
                    saved += 1
                index += 1
        if saved == 0:
            raise FalApiError(
                _ARCHIVE_MODEL,
                "No frames could be decoded from the video. "
                "Check the input video and the every_nth setting.",
            )
        logger.info("Extracted %d frame(s) from %s", saved, video_path)
        return zip_path
    except Exception:
        _safe_unlink(zip_path)
        raise
    finally:
        capture.release()


class FalImagesToZipURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Images to package as a training dataset zip (image_0.png, image_1.png, ...).",
                    },
                ),
            },
            "optional": {
                "captions": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional captions, one per line (blank lines allowed). "
                        "Line count must match the image batch size, or leave empty for no captions. "
                        "Written as image_0.txt, image_1.txt, ... next to each image.",
                    },
                ),
                "name_prefix": (
                    "STRING",
                    {
                        "default": "image",
                        "tooltip": "File name prefix inside the zip (e.g. 'image' -> image_0.png).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_url",)
    FUNCTION = "create_zip_url"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Zips an IMAGE batch (with optional per-image captions) and uploads it to fal.ai. "
        "Feed the URL directly into the LoRA trainer nodes' images_data_url input."
    )

    def create_zip_url(self, images, captions="", name_prefix="image"):
        caption_lines = _split_caption_lines(captions)
        zip_path = ArchiveUtils.zip_images(
            images, captions=caption_lines, name_prefix=name_prefix
        )
        return (ArchiveUtils.upload_zip(zip_path),)


class FalFolderToZipURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a local folder whose files will be zipped and uploaded.",
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also include files from subfolders (hidden entries are always skipped).",
                    },
                ),
                "extensions": (
                    "STRING",
                    {
                        "default": ".png,.jpg,.jpeg,.webp,.txt",
                        "tooltip": "Comma-separated list of file extensions to include. Empty includes all files.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_url",)
    FUNCTION = "create_zip_url"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Zips a local folder and uploads it to fal.ai, returning the zip URL."

    def create_zip_url(self, folder_path, recursive=False, extensions=".png,.jpg,.jpeg,.webp,.txt"):
        extension_list = [part.strip() for part in extensions.split(",") if part.strip()]
        zip_path = ArchiveUtils.zip_folder(
            folder_path,
            include_extensions=extension_list or None,
            recursive=recursive,
        )
        return (ArchiveUtils.upload_zip(zip_path),)


class FalVideoToFrameDatasetZip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Video to sample frames from for a training dataset.",
                    },
                ),
                "every_nth": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Keep one frame out of every N decoded frames.",
                    },
                ),
                "max_frames": (
                    "INT",
                    {
                        "default": 200,
                        "min": 1,
                        "max": 2000,
                        "step": 1,
                        "tooltip": "Stop after this many frames have been saved.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_url",)
    FUNCTION = "create_zip_url"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Samples frames from a video (every Nth frame, up to max_frames), zips them as PNGs, "
        "uploads the zip to fal.ai, and returns the URL."
    )

    def create_zip_url(self, video, every_nth=10, max_frames=200):
        local_path, is_temp = _video_to_local_path(video)
        try:
            zip_path = _extract_frames_to_zip(local_path, every_nth, max_frames)
        finally:
            if is_temp:
                _safe_unlink(local_path)
        return (ArchiveUtils.upload_zip(zip_path),)


class FalBatchCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Images to caption, one caption per frame.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image for LoRA training in one dense sentence.",
                        "multiline": True,
                        "tooltip": "Instruction sent to the vision model for each image.",
                    },
                ),
                "model": (
                    [
                        "google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4o",
                        "custom",
                    ],
                    {
                        "default": "google/gemini-2.5-flash",
                        "tooltip": "Vision model to use. Select 'custom' to type any OpenRouter model id "
                        "in custom_model_name.",
                    },
                ),
            },
            "optional": {
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "OpenRouter model id used when model is set to 'custom'.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "caption_images"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Captions each image in a batch with a fal VLM (concurrently, order preserved) and returns "
        "one caption per line — wire straight into FalImagesToZipURL's captions input."
    )

    def caption_images(
        self,
        images,
        prompt="Describe this image for LoRA training in one dense sentence.",
        model="google/gemini-2.5-flash",
        custom_model_name="",
    ):
        if model == "custom":
            if not custom_model_name or not custom_model_name.strip():
                raise FalApiError(
                    _VISION_ENDPOINT,
                    "custom_model_name is required when model is set to 'custom'.",
                )
            model = custom_model_name.strip()

        image_urls = ImageUtils.prepare_images(images)
        if not image_urls:
            raise FalApiError(
                _VISION_ENDPOINT, "No images provided to caption. Connect an IMAGE batch."
            )

        def caption_one(image_url: str) -> str:
            arguments = {
                "model": model,
                "prompt": prompt,
                "image_urls": [image_url],
                "stream": False,
            }
            result = ApiHandler.submit_and_get_result(_VISION_ENDPOINT, arguments)
            # Captions are joined by newline, so flatten any multiline output.
            return str(result["output"]).replace("\r", " ").replace("\n", " ").strip()

        with ThreadPoolExecutor(max_workers=_MAX_CAPTION_WORKERS) as executor:
            futures = [executor.submit(caption_one, url) for url in image_urls]

        captions: list[str] = []
        failure_count = 0
        for index, future in enumerate(futures):
            try:
                captions = [*captions, future.result()]
            except Exception as exc:
                # a user Cancel raised inside a worker must stop the node,
                # not silently become an empty caption
                if exc.__class__.__name__ == "InterruptProcessingException":
                    raise
                logger.warning("Caption for image %d failed: %s", index, exc)
                captions = [*captions, ""]
                failure_count += 1

        if failure_count == len(image_urls):
            raise FalApiError(
                _VISION_ENDPOINT,
                f"All {len(image_urls)} caption request(s) failed. "
                "Check the model id, your fal API key, and the queue logs above.",
            )
        return ("\n".join(captions),)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FalImagesToZipURL_fal": FalImagesToZipURL,
    "FalFolderToZipURL_fal": FalFolderToZipURL,
    "FalVideoToFrameDatasetZip_fal": FalVideoToFrameDatasetZip,
    "FalBatchCaption_fal": FalBatchCaption,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalImagesToZipURL_fal": "Images → Training ZIP URL (fal)",
    "FalFolderToZipURL_fal": "Folder → ZIP URL (fal)",
    "FalVideoToFrameDatasetZip_fal": "Video → Frame Dataset ZIP URL (fal)",
    "FalBatchCaption_fal": "Batch Caption Images (fal VLM)",
}
