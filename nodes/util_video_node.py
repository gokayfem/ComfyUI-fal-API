"""Local video utility nodes: frame extraction, trim, concat, mux, audio extraction.

These nodes run entirely locally (cv2/PyAV) — no fal.ai API calls — and are
meant to glue video-generation workflows together (e.g. grab the last frame of
a clip and feed it into an image-to-video node to extend the video).
"""

from __future__ import annotations

import os
import tempfile
from fractions import Fraction
from typing import Any

import numpy as np
import torch

from .fal_utils import FalApiError, MediaUtils, logger

_CATEGORY = "FAL/Utils/Video"

_CHUNK_SIZE = 1 << 20  # 1 MiB
_AV_TIME_BASE = 1_000_000  # PyAV container.seek() offset units (microseconds)
_AAC_FRAME_SIZE = 1024  # samples per AAC frame
_TIME_EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_unlink(path: str | None) -> None:
    """Delete a temp file, ignoring errors."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _import_cv2(node_name: str) -> Any:
    """Import cv2 lazily with a clear error when missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise FalApiError(
            node_name, "OpenCV is required for this node — pip install opencv-python"
        ) from exc


def _import_av(node_name: str) -> Any:
    """Import PyAV lazily with a clear error when missing."""
    try:
        import av

        return av
    except ImportError as exc:
        raise FalApiError(node_name, "PyAV is required for this node — pip install av") from exc


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


def _wrap_local_video(path: str, node_name: str) -> Any:
    """Wrap a local video file as a ComfyUI VIDEO object."""
    video_cls = _resolve_video_from_file()
    if video_cls is None:
        raise FalApiError(
            node_name,
            "comfy_api VideoFromFile is unavailable; update ComfyUI to a version "
            "that provides comfy_api to use VIDEO outputs.",
        )
    return video_cls(path)


def _new_temp_path(suffix: str) -> str:
    """Create an empty named temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix="fal_util_video_", delete=False) as temp_file:
        return temp_file.name


def _spool_stream_to_temp(source: Any, node_name: str) -> str:
    """Write a readable stream to a temp .mp4 file and return its path."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="fal_util_video_", delete=False) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = source.read(_CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)
        return temp_path
    except Exception as exc:
        _safe_unlink(temp_path)
        raise FalApiError(node_name, f"Failed to read video stream: {exc}") from exc


def _video_input_to_path(video: Any, node_name: str) -> tuple[str, bool]:
    """Resolve a VIDEO input (or path/URL string) to a local file path.

    Returns (path, cleanup_needed). cleanup_needed is True when the path is a
    temp file created here that the caller must delete when done.
    """
    if video is None:
        raise FalApiError(node_name, "No video input provided")

    source = video.get_stream_source() if hasattr(video, "get_stream_source") else video

    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return MediaUtils.download_url_to_temp(source, ".mp4"), True
        if os.path.isfile(source):
            return source, False
        raise FalApiError(node_name, f"Video path does not exist: {source}")

    if hasattr(source, "read"):
        return _spool_stream_to_temp(source, node_name), True

    raise FalApiError(
        node_name,
        f"Unsupported video input of type {type(video).__name__}; expected a "
        "VIDEO object, a local file path, or an http(s) URL string.",
    )


def _add_stream_from_template(output: Any, template: Any) -> Any:
    """Add an output stream copying the template's codec parameters."""
    if hasattr(output, "add_stream_from_template"):
        return output.add_stream_from_template(template)
    return output.add_stream(template=template)


def _stream_duration_seconds(container: Any, stream: Any) -> float:
    """Best-effort duration (seconds) of a stream, falling back to container."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / _AV_TIME_BASE
    return 0.0


def _encode_audio_array(
    av: Any, output: Any, stream: Any, layout: str, sample_rate: int, samples: np.ndarray, start_index: int
) -> int:
    """Encode a planar float32 (C, T) array as AAC frames; returns next sample index."""
    total = samples.shape[1]
    for offset in range(0, total, _AAC_FRAME_SIZE):
        chunk = np.ascontiguousarray(samples[:, offset : offset + _AAC_FRAME_SIZE])
        frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
        frame.sample_rate = sample_rate
        frame.pts = start_index + offset
        for packet in stream.encode(frame):
            output.mux(packet)
    return start_index + total


def _pad_or_truncate(samples: np.ndarray, needed: int) -> np.ndarray:
    """Pad a planar (C, T) array with silence, or truncate, to exactly `needed` samples."""
    if samples.shape[1] >= needed:
        return samples[:, :needed]
    pad = np.zeros((samples.shape[0], needed - samples.shape[1]), dtype=np.float32)
    return np.concatenate([samples, pad], axis=1)


def _normalize_audio_frame(array: np.ndarray, channels: int) -> np.ndarray:
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


def _waveform_to_planar(audio: Any, node_name: str) -> tuple[np.ndarray, int]:
    """Convert a ComfyUI AUDIO dict to (planar float32 (C, T) with C in {1, 2}, sample_rate)."""
    try:
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
    except (KeyError, TypeError) as exc:
        raise FalApiError(
            node_name, "Expected an AUDIO dict with 'waveform' and 'sample_rate'"
        ) from exc
    if not isinstance(waveform, torch.Tensor):
        raise FalApiError(node_name, "AUDIO 'waveform' must be a torch tensor")

    tensor = waveform.detach().cpu().to(torch.float32)
    if tensor.ndim == 3:
        tensor = tensor[0]  # (B, C, T) -> (C, T)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise FalApiError(node_name, f"AUDIO waveform has unsupported shape {tuple(waveform.shape)}")

    array = tensor.clamp(-1.0, 1.0).numpy()
    if array.shape[0] > 2:
        logger.warning("%s: waveform has %d channels; keeping the first two", node_name, array.shape[0])
        array = array[:2]
    return np.ascontiguousarray(array), sample_rate


# ---------------------------------------------------------------------------
# FalExtractFrames
# ---------------------------------------------------------------------------


def _bgr_frames_to_tensor(frames: list[np.ndarray], cv2: Any) -> torch.Tensor:
    """Convert BGR uint8 frames to a (N, H, W, 3) float32 RGB tensor in 0-1."""
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    stacked = np.stack(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def _frame_via_seek(cap: Any, cv2: Any, index: int) -> np.ndarray | None:
    """Seek to a frame index and read it; returns None when the seek misbehaves."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def _scan_frames(cap: Any, cv2: Any, stop_after: int | None = None) -> tuple[np.ndarray | None, int]:
    """Sequentially decode from frame 0; returns (last frame seen, frames read).

    Stops after reading `stop_after + 1` frames when `stop_after` is given.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    last: np.ndarray | None = None
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        last = frame
        count += 1
        if stop_after is not None and count > stop_after:
            break
    return last, count


class FalExtractFrames:
    """Extract frames from a video as IMAGE outputs (local decode, no API call)."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "extract_frames"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Decode a video locally and extract frames. Mode 'last' grabs the final "
        "frame — feed it into an image-to-video node to extend/continue a video."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Video to decode. Tip: use mode 'last' to grab the final frame and feed it into an image-to-video node to extend the video."}),
                "mode": (["first", "last", "nth", "every_nth"], {"default": "last", "tooltip": "first/last: single frame. nth: the n-th frame (1-based). every_nth: every n-th frame as a batch, capped at max_frames."}),
                "n": ("INT", {"default": 1, "min": 1, "max": 1_000_000, "tooltip": "Frame index (1-based) for mode 'nth'; step size for 'every_nth'. Ignored otherwise."}),
                "max_frames": ("INT", {"default": 64, "min": 1, "max": 1024, "tooltip": "Maximum number of frames returned by mode 'every_nth'; ignored for other modes."}),
            },
        }

    def extract_frames(self, video: Any, mode: str, n: int, max_frames: int) -> tuple[torch.Tensor, int]:
        node_name = "FalExtractFrames"
        cv2 = _import_cv2(node_name)
        path, cleanup = _video_input_to_path(video, node_name)
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FalApiError(node_name, f"OpenCV could not open video: {path}")
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if mode == "every_nth":
                frames, frame_count = self._extract_every_nth(cap, cv2, n, max_frames, reported)
            else:
                frame, frame_count = self._extract_single(cap, cv2, mode, n, reported, node_name)
                frames = [frame]

            if not frames or frames[0] is None:
                raise FalApiError(node_name, f"No frames could be decoded from {path}")
            return (_bgr_frames_to_tensor(frames, cv2), frame_count)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("FalExtractFrames failed: %s", exc)
            raise FalApiError(node_name, f"Frame extraction failed: {exc}") from exc
        finally:
            if cap is not None:
                cap.release()
            if cleanup:
                _safe_unlink(path)

    @staticmethod
    def _extract_every_nth(
        cap: Any, cv2: Any, step: int, max_frames: int, reported: int
    ) -> tuple[list[np.ndarray], int]:
        """Sequentially collect every `step`-th frame, capped at max_frames."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        frames: list[np.ndarray] = []
        count = 0
        reached_eof = False
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                reached_eof = True
                break
            if count % step == 0 and len(frames) < max_frames:
                frames.append(frame)
            count += 1
            if len(frames) >= max_frames and reported > 0:
                break  # early stop: reported count stands in for the true total
        frame_count = count if reached_eof or reported <= 0 else reported
        return frames, frame_count

    @staticmethod
    def _extract_single(
        cap: Any, cv2: Any, mode: str, n: int, reported: int, node_name: str
    ) -> tuple[np.ndarray | None, int]:
        """Extract a single frame for modes first/last/nth."""
        if mode == "first":
            target = 0
        elif mode == "nth":
            target = n - 1
            if reported > 0 and target >= reported:
                logger.warning("%s: frame %d beyond end (%d frames); using last frame", node_name, n, reported)
                target = reported - 1
        elif mode == "last":
            target = max(reported - 1, 0)
        else:
            raise FalApiError(node_name, f"Unknown mode: {mode}")

        frame: np.ndarray | None = None
        if reported > 0:
            # Fast path: direct seek (some codecs mis-seek; fall back below).
            frame = _frame_via_seek(cap, cv2, target)
        if frame is not None:
            return frame, reported

        # Sequential fallback: decode from the start.
        if mode == "last":
            frame, count = _scan_frames(cap, cv2)
            return frame, count
        frame, read = _scan_frames(cap, cv2, stop_after=target)
        if read > target:
            # Reached the target; total count comes from metadata or a full scan.
            frame_count = reported if reported > 0 else _scan_frames(cap, cv2)[1]
            return frame, frame_count
        # Hit EOF early: `frame` is the last decodable frame, `read` the true count.
        return frame, read


# ---------------------------------------------------------------------------
# FalTrimVideo
# ---------------------------------------------------------------------------


def _find_seek_start(av: Any, container: Any, anchor: Any, start: float) -> float:
    """Seek near `start` and return the timestamp of the first packet (keyframe snap)."""
    if start <= 0:
        return 0.0
    container.seek(int(start * _AV_TIME_BASE), backward=True, any_frame=False)
    actual_start = start
    for packet in container.demux(anchor):
        if packet.pts is None:
            continue
        actual_start = float(packet.pts * packet.time_base)
        break
    container.seek(int(start * _AV_TIME_BASE), backward=True, any_frame=False)
    return actual_start


def _remux_trim(av: Any, in_path: str, out_path: str, start: float, end: float | None, node_name: str) -> None:
    """Copy packets between timestamps into a new mp4 without re-encoding."""
    with av.open(in_path) as container, av.open(out_path, mode="w") as output:
        video_in = container.streams.video[0] if container.streams.video else None
        audio_in = container.streams.audio[0] if container.streams.audio else None
        selected = [stream for stream in (video_in, audio_in) if stream is not None]
        if not selected:
            raise FalApiError(node_name, "Input has no video or audio streams")

        out_streams = {stream.index: _add_stream_from_template(output, stream) for stream in selected}
        anchor = video_in if video_in is not None else audio_in
        actual_start = _find_seek_start(av, container, anchor, start)
        if end is not None and end <= actual_start + _TIME_EPS:
            raise FalApiError(
                node_name,
                f"Trim range is empty: start snapped to keyframe at {actual_start:.3f}s, end is {end:.3f}s",
            )

        offsets: dict[int, int] = {}
        done = dict.fromkeys(out_streams, False)
        kept = 0
        for packet in container.demux(selected):
            if packet.pts is None:
                continue
            index = packet.stream.index
            if done.get(index, True):
                continue
            time = float(packet.pts * packet.time_base)
            if time < actual_start - _TIME_EPS:
                continue
            if end is not None and time >= end - _TIME_EPS:
                done[index] = True
                if all(done.values()):
                    break
                continue
            if index not in offsets:
                offsets[index] = packet.dts if packet.dts is not None else packet.pts
            offset = offsets[index]
            packet.pts -= offset
            if packet.dts is not None:
                packet.dts -= offset
            packet.stream = out_streams[index]
            output.mux(packet)
            kept += 1

        if kept == 0:
            raise FalApiError(node_name, f"Trim produced no packets (start {start:.3f}s may be past the end)")


class FalTrimVideo:
    """Trim a video to [start, end] seconds by remuxing (no re-encode)."""

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "trim_video"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Trim a video without re-encoding by copying packets between timestamps. "
        "Fast and lossless, but the start cut snaps to the nearest earlier keyframe."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Video to trim (video + audio tracks are kept)."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100_000.0, "step": 0.1, "tooltip": "Trim start in seconds. Cuts snap to the nearest earlier keyframe (no re-encode)."}),
                "end_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100_000.0, "step": 0.1, "tooltip": "Trim end in seconds; 0 = to the end of the video."}),
            },
        }

    def trim_video(self, video: Any, start_seconds: float, end_seconds: float) -> tuple[Any, str]:
        node_name = "FalTrimVideo"
        av = _import_av(node_name)
        end = end_seconds if end_seconds > 0 else None
        if end is not None and end <= start_seconds:
            raise FalApiError(node_name, f"end_seconds ({end:.3f}) must be greater than start_seconds ({start_seconds:.3f})")
        path, cleanup = _video_input_to_path(video, node_name)
        out_path = _new_temp_path(".mp4")
        try:
            _remux_trim(av, path, out_path, start_seconds, end, node_name)
            # NOTE: out_path is deliberately kept — VideoFromFile reads it lazily.
            return (_wrap_local_video(out_path, node_name), out_path)
        except FalApiError:
            _safe_unlink(out_path)
            raise
        except Exception as exc:
            _safe_unlink(out_path)
            logger.error("FalTrimVideo failed: %s", exc)
            raise FalApiError(node_name, f"Trim failed: {exc}") from exc
        finally:
            if cleanup:
                _safe_unlink(path)


# ---------------------------------------------------------------------------
# FalConcatVideos
# ---------------------------------------------------------------------------


def _probe_video(av: Any, path: str, node_name: str) -> dict[str, Any]:
    """Probe a clip for resolution, fps, audio presence, and audio rate."""
    with av.open(path) as container:
        if not container.streams.video:
            raise FalApiError(node_name, f"Input has no video stream: {path}")
        stream = container.streams.video[0]
        fps = stream.average_rate or stream.guessed_rate
        if not fps or fps <= 0:
            fps = Fraction(30, 1)
        audio_rate = None
        if container.streams.audio:
            audio_rate = int(container.streams.audio[0].rate or 44100)
        return {
            "width": max(2, stream.width - stream.width % 2),
            "height": max(2, stream.height - stream.height % 2),
            "fps": Fraction(fps),
            "audio_rate": audio_rate,
        }


def _encode_video_frame(output: Any, stream: Any, frame: Any, width: int, height: int, time_base: Fraction, index: int) -> None:
    """Scale/convert one decoded frame and encode it at the given frame index."""
    scaled = frame.reformat(width=width, height=height, format="yuv420p")
    scaled.pts = index
    scaled.time_base = time_base
    for packet in stream.encode(scaled):
        output.mux(packet)


def _append_clip_video(
    av: Any, output: Any, stream: Any, path: str, width: int, height: int, fps: Fraction, start_index: int, node_name: str
) -> int:
    """Decode a clip, resample to target fps/size, encode; returns frames emitted."""
    time_base = Fraction(1, 1) / fps
    step = 1.0 / float(fps)
    emitted = 0
    with av.open(path) as container:
        next_time = 0.0
        last = None
        for frame in container.decode(container.streams.video[0]):
            time = frame.time if frame.time is not None else next_time
            while last is not None and time > next_time + _TIME_EPS:
                _encode_video_frame(output, stream, last, width, height, time_base, start_index + emitted)
                emitted += 1
                next_time += step
            last = frame
        if last is not None:
            _encode_video_frame(output, stream, last, width, height, time_base, start_index + emitted)
            emitted += 1
    if emitted == 0:
        raise FalApiError(node_name, f"No video frames decoded from {path}")
    return emitted


def _clip_audio_samples(av: Any, path: str, rate: int, needed: int) -> np.ndarray:
    """Decode+resample a clip's audio to stereo float32 (2, needed); silence when absent."""
    with av.open(path) as container:
        if not container.streams.audio:
            return np.zeros((2, needed), dtype=np.float32)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(container.streams.audio[0]):
            frame.pts = None  # let the resampler track timestamps itself
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not chunks:
        return np.zeros((2, needed), dtype=np.float32)
    samples = np.concatenate(chunks, axis=1).astype(np.float32)
    return _pad_or_truncate(samples, needed)


class FalConcatVideos:
    """Concatenate 2-4 videos by re-encoding to the first clip's resolution and fps."""

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "concat_videos"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Concatenate up to 4 videos. All clips are re-encoded (h264 crf 18) and scaled to "
        "video_1's resolution and fps, so mismatched codecs/sizes are fine. Audio: the output "
        "gets a stereo AAC track when any input has audio; inputs without audio contribute silence."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_1": ("VIDEO", {"tooltip": "First clip; its resolution and fps define the output format."}),
                "video_2": ("VIDEO", {"tooltip": "Second clip, scaled to match video_1."}),
            },
            "optional": {
                "video_3": ("VIDEO", {"tooltip": "Optional third clip."}),
                "video_4": ("VIDEO", {"tooltip": "Optional fourth clip."}),
            },
        }

    def concat_videos(
        self, video_1: Any, video_2: Any, video_3: Any = None, video_4: Any = None
    ) -> tuple[Any, str]:
        node_name = "FalConcatVideos"
        av = _import_av(node_name)

        inputs = [video for video in (video_1, video_2, video_3, video_4) if video is not None]
        resolved: list[tuple[str, bool]] = []
        out_path = _new_temp_path(".mp4")
        try:
            resolved = [_video_input_to_path(video, node_name) for video in inputs]
            paths = [path for path, _ in resolved]
            probes = [_probe_video(av, path, node_name) for path in paths]

            target = probes[0]
            width, height, fps = target["width"], target["height"], target["fps"]
            audio_rates = [probe["audio_rate"] for probe in probes if probe["audio_rate"]]
            audio_rate = audio_rates[0] if audio_rates else None

            with av.open(out_path, mode="w") as output:
                video_out = output.add_stream("libx264", rate=fps, options={"crf": "18", "preset": "veryfast"})
                video_out.width = width
                video_out.height = height
                video_out.pix_fmt = "yuv420p"
                audio_out = None
                if audio_rate is not None:
                    audio_out = output.add_stream("aac", rate=audio_rate)
                    audio_out.layout = "stereo"

                frame_index = 0
                sample_index = 0
                for path in paths:
                    emitted = _append_clip_video(av, output, video_out, path, width, height, fps, frame_index, node_name)
                    frame_index += emitted
                    if audio_out is not None:
                        needed = round(emitted / float(fps) * audio_rate)
                        samples = _clip_audio_samples(av, path, audio_rate, needed)
                        sample_index = _encode_audio_array(av, output, audio_out, "stereo", audio_rate, samples, sample_index)

                for packet in video_out.encode(None):
                    output.mux(packet)
                if audio_out is not None:
                    for packet in audio_out.encode(None):
                        output.mux(packet)

            # NOTE: out_path is deliberately kept — VideoFromFile reads it lazily.
            return (_wrap_local_video(out_path, node_name), out_path)
        except FalApiError:
            _safe_unlink(out_path)
            raise
        except Exception as exc:
            _safe_unlink(out_path)
            logger.error("FalConcatVideos failed: %s", exc)
            raise FalApiError(node_name, f"Concat failed: {exc}") from exc
        finally:
            for path, cleanup in resolved:
                if cleanup:
                    _safe_unlink(path)


# ---------------------------------------------------------------------------
# FalMuxAudioVideo
# ---------------------------------------------------------------------------


class FalMuxAudioVideo:
    """Mux an AUDIO waveform onto a video, replacing any existing audio track."""

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "mux_audio_video"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Attach an AUDIO input to a video as an AAC track, replacing any existing audio. "
        "Video packets are copied without re-encoding."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Video track; copied without re-encoding. Existing audio is replaced."}),
                "audio": ("AUDIO", {"tooltip": "Audio to attach, encoded as AAC at its own sample rate."}),
                "duration_policy": (["video", "shortest"], {"default": "video", "tooltip": "video: keep full video; audio is padded with silence or truncated to match. shortest: cut the output at whichever track ends first."}),
            },
        }

    def mux_audio_video(self, video: Any, audio: Any, duration_policy: str) -> tuple[Any, str]:
        node_name = "FalMuxAudioVideo"
        av = _import_av(node_name)
        samples, sample_rate = _waveform_to_planar(audio, node_name)
        layout = "mono" if samples.shape[0] == 1 else "stereo"
        audio_duration = samples.shape[1] / float(sample_rate)

        path, cleanup = _video_input_to_path(video, node_name)
        out_path = _new_temp_path(".mp4")
        try:
            with av.open(path) as container:
                if not container.streams.video:
                    raise FalApiError(node_name, "Input has no video stream")
                video_in = container.streams.video[0]
                video_duration = _stream_duration_seconds(container, video_in)
                if video_duration <= 0:
                    raise FalApiError(node_name, "Could not determine the video duration")
                target = video_duration if duration_policy == "video" else min(video_duration, audio_duration)

                with av.open(out_path, mode="w") as output:
                    video_out = _add_stream_from_template(output, video_in)
                    audio_out = output.add_stream("aac", rate=sample_rate)
                    audio_out.layout = layout

                    for packet in container.demux(video_in):
                        if packet.dts is None:
                            continue
                        if duration_policy == "shortest" and float(packet.dts * packet.time_base) >= target - _TIME_EPS:
                            break
                        packet.stream = video_out
                        output.mux(packet)

                    needed = round(target * sample_rate)
                    _encode_audio_array(
                        av, output, audio_out, layout, sample_rate, _pad_or_truncate(samples, needed), 0
                    )
                    for packet in audio_out.encode(None):
                        output.mux(packet)

            # NOTE: out_path is deliberately kept — VideoFromFile reads it lazily.
            return (_wrap_local_video(out_path, node_name), out_path)
        except FalApiError:
            _safe_unlink(out_path)
            raise
        except Exception as exc:
            _safe_unlink(out_path)
            logger.error("FalMuxAudioVideo failed: %s", exc)
            raise FalApiError(node_name, f"Mux failed: {exc}") from exc
        finally:
            if cleanup:
                _safe_unlink(path)


# ---------------------------------------------------------------------------
# FalVideoToAudio
# ---------------------------------------------------------------------------


class FalVideoToAudio:
    """Extract the audio track of a video as a ComfyUI AUDIO output."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "video_to_audio"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Extract a video's audio track as an AUDIO output at its original sample rate."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Video whose audio track should be extracted."}),
            },
        }

    def video_to_audio(self, video: Any) -> tuple[dict[str, Any]]:
        node_name = "FalVideoToAudio"
        av = _import_av(node_name)
        path, cleanup = _video_input_to_path(video, node_name)
        try:
            with av.open(path) as container:
                if not container.streams.audio:
                    raise FalApiError(node_name, "video has no audio track")
                stream = container.streams.audio[0]
                sample_rate = int(stream.rate or 44100)
                channels = int(getattr(stream, "channels", 1) or 1)
                chunks = [
                    _normalize_audio_frame(frame.to_ndarray(), channels)
                    for frame in container.decode(stream)
                ]
            if not chunks:
                raise FalApiError(node_name, "No audio frames could be decoded")
            waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
            return ({"waveform": waveform, "sample_rate": sample_rate},)
        except FalApiError:
            raise
        except Exception as exc:
            logger.error("FalVideoToAudio failed: %s", exc)
            raise FalApiError(node_name, f"Audio extraction failed: {exc}") from exc
        finally:
            if cleanup:
                _safe_unlink(path)


NODE_CLASS_MAPPINGS = {
    "FalExtractFrames_fal": FalExtractFrames,
    "FalTrimVideo_fal": FalTrimVideo,
    "FalConcatVideos_fal": FalConcatVideos,
    "FalMuxAudioVideo_fal": FalMuxAudioVideo,
    "FalVideoToAudio_fal": FalVideoToAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalExtractFrames_fal": "Extract Frames (fal)",
    "FalTrimVideo_fal": "Trim Video (fal)",
    "FalConcatVideos_fal": "Concat Videos (fal)",
    "FalMuxAudioVideo_fal": "Mux Audio + Video (fal)",
    "FalVideoToAudio_fal": "Video → Audio (fal)",
}
