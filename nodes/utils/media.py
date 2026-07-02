"""Video/audio helpers for ComfyUI-fal-API (download, decode, upload)."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch

from .errors import FalApiError
from .images import ImageUtils
from .logger import logger

_DOWNLOAD_TIMEOUT = (10, 600)
_CHUNK_SIZE = 1 << 20  # 1 MiB
_video_warning = {"emitted": False, "lock": threading.Lock()}


def _safe_unlink(path: str) -> None:
    """Delete a temp file, ignoring errors."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _suffix_from_url(url: str, default: str) -> str:
    """Derive a file suffix from a URL path, falling back to a default."""
    suffix = os.path.splitext(urlparse(url).path)[1]
    return suffix if suffix else default


def _resolve_video_from_file() -> type | None:
    """Locate ComfyUI's VideoFromFile class across API layouts."""
    try:
        from comfy_api.input_impl import VideoFromFile

        return VideoFromFile
    except ImportError:
        pass
    try:
        from comfy_api.latest import input_impl

        return getattr(input_impl, "VideoFromFile", None)
    except ImportError:
        return None


def _warn_video_unavailable_once() -> None:
    """Warn (once) that ComfyUI VIDEO output support is unavailable."""
    with _video_warning["lock"]:
        if not _video_warning["emitted"]:
            _video_warning["emitted"] = True
            logger.warning(
                "comfy_api VideoFromFile is unavailable; VIDEO outputs will be "
                "None. Update ComfyUI to a version that provides comfy_api."
            )


def _normalize_av_frame(array: np.ndarray, channels: int) -> np.ndarray:
    """Normalize a PyAV audio frame array to float32 with shape (C, N)."""
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32)

    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.shape[0] == 1 and channels > 1:
        # Packed/interleaved format: (1, N * C) -> (C, N)
        array = array.reshape(-1, channels).T
    return array


def _load_audio_with_av(path: str) -> tuple[torch.Tensor, int]:
    """Decode audio with PyAV; returns (waveform (1, C, T) float32, rate)."""
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or 44100)
        channels = int(getattr(stream, "channels", 1) or 1)
        frames = [
            _normalize_av_frame(frame.to_ndarray(), channels)
            for frame in container.decode(stream)
        ]

    if not frames:
        raise FalApiError("audio-decode", f"No audio frames decoded from {path}")
    waveform = torch.from_numpy(np.concatenate(frames, axis=1))
    return waveform.unsqueeze(0), sample_rate


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Decode an audio file to (waveform (1, C, T) float32, sample_rate)."""
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        return waveform.to(torch.float32).unsqueeze(0), int(sample_rate)
    except ImportError:
        pass

    try:
        return _load_audio_with_av(path)
    except ImportError as exc:
        raise FalApiError(
            "audio-decode",
            "Decoding audio requires torchaudio or av (PyAV); neither is "
            "installed. Install one of them (e.g. 'pip install torchaudio').",
        ) from exc


def _save_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a (C, T) float32 waveform as WAV (torchaudio, else stdlib PCM16)."""
    try:
        import torchaudio

        torchaudio.save(path, waveform, sample_rate)
        return
    except ImportError:
        pass

    import wave

    clipped = np.clip(waveform.numpy(), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.T.reshape(-1).tobytes())


def _stream_to_temp_file(source: Any, suffix: str) -> str:
    """Write a readable stream to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name
        while True:
            chunk = source.read(_CHUNK_SIZE)
            if not chunk:
                break
            temp_file.write(chunk)
    return temp_path


class MediaUtils:
    """Utility functions for video/audio download, conversion, and upload."""

    @staticmethod
    def download_url_to_temp(url: str, suffix: str) -> str:
        """Stream a URL to a temp file and return its local path."""
        temp_path: str | None = None
        try:
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp_file:
                    temp_path = temp_file.name
                    for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            temp_file.write(chunk)
            return temp_path
        except Exception as exc:
            if temp_path is not None:
                _safe_unlink(temp_path)
            logger.error("Failed to download %s: %s", url, exc)
            raise FalApiError(
                "media-download", f"Failed to download {url}: {exc}"
            ) from exc

    @staticmethod
    def video_from_url(url: str) -> Any | None:
        """Download a video URL and wrap it as a ComfyUI VIDEO object."""
        video_cls = _resolve_video_from_file()
        if video_cls is None:
            _warn_video_unavailable_once()
            return None
        # NOTE: the temp file is deliberately not unlinked here — VideoFromFile
        # reads the path lazily (e.g. when a downstream save node consumes it),
        # so deleting early would break playback. The OS temp dir reclaims it.
        local_path = MediaUtils.download_url_to_temp(
            url, _suffix_from_url(url, default=".mp4")
        )
        return video_cls(local_path)

    @staticmethod
    def audio_from_url(url: str) -> dict[str, Any]:
        """Download and decode audio into a ComfyUI AUDIO dict.

        Returns {"waveform": float32 tensor (1, C, T), "sample_rate": int}.
        """
        local_path = MediaUtils.download_url_to_temp(
            url, _suffix_from_url(url, default=".wav")
        )
        try:
            waveform, sample_rate = _load_audio(local_path)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("Failed to decode audio from %s: %s", url, exc)
            raise FalApiError(
                "audio-decode", f"Failed to decode audio from {url}: {exc}"
            ) from exc
        finally:
            _safe_unlink(local_path)

    @staticmethod
    def upload_video(video: Any) -> str:
        """Upload a ComfyUI VIDEO input (or path/url string) and return a URL."""
        if isinstance(video, str):
            return video if _is_http_url(video) else ImageUtils.upload_file(video)

        source = (
            video.get_stream_source()
            if hasattr(video, "get_stream_source")
            else video
        )
        if isinstance(source, str) and _is_http_url(source):
            return source
        if hasattr(source, "read"):
            temp_path = _stream_to_temp_file(source, suffix=".mp4")
            try:
                return ImageUtils.upload_file(temp_path)
            finally:
                _safe_unlink(temp_path)
        return ImageUtils.upload_file(source)

    @staticmethod
    def upload_audio(audio: Any) -> str:
        """Upload a ComfyUI AUDIO dict (or path/url string) and return a URL."""
        if isinstance(audio, str):
            return audio if _is_http_url(audio) else ImageUtils.upload_file(audio)

        try:
            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])
        except (KeyError, TypeError) as exc:
            raise FalApiError(
                "audio-upload",
                "Expected an AUDIO dict with 'waveform' and 'sample_rate'",
            ) from exc

        tensor = waveform.detach().cpu().to(torch.float32)
        if tensor.ndim == 3:
            tensor = tensor[0]  # (1, C, T) -> (C, T)

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            _save_wav(temp_path, tensor, sample_rate)
            return ImageUtils.upload_file(temp_path)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("Failed to save/upload audio: %s", exc)
            raise FalApiError(
                "audio-upload", f"Failed to save/upload audio: {exc}"
            ) from exc
        finally:
            if temp_path is not None:
                _safe_unlink(temp_path)
