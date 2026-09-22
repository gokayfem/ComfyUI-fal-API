"""Focused tests for the curated Seedance 2.5 video editing node."""

from __future__ import annotations

import math
import sys


class _URLVideo:
    def __init__(self, url: str):
        self.url = url

    def get_stream_source(self):
        return self.url


def _node_and_module(pack):
    node_cls = pack.NODE_CLASS_MAPPINGS["Seedance25VideoToVideo_fal"]
    return node_cls, sys.modules[node_cls.__module__]


def test_seedance25_video_to_video_schema_and_registration(pack):
    node_cls, _module = _node_and_module(pack)
    inputs = node_cls.INPUT_TYPES()

    assert inputs["required"]["video"][0] == "VIDEO"
    assert inputs["required"]["prompt"][0] == "STRING"
    assert set(inputs["optional"]) == {
        "resolution",
        "generate_audio",
        "bitrate_mode",
        "seed",
        "force_rerun",
    }
    assert node_cls.RETURN_TYPES == ("VIDEO", "STRING")
    assert node_cls.RETURN_NAMES == ("video", "video_url")
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["Seedance25VideoToVideo_fal"] == (
        "Seedance 2.5 Video-to-Video (fal)"
    )


def test_seedance25_edit_payload_outputs_and_force_rerun(pack, monkeypatch):
    node_cls, module = _node_and_module(pack)
    submitted = {}
    native_output = object()

    monkeypatch.setattr(
        module.MediaUtils,
        "upload_video",
        staticmethod(lambda _video: "https://fal.media/input.mp4"),
    )
    monkeypatch.setattr(
        module.MediaUtils,
        "video_from_url",
        staticmethod(lambda url: native_output if url.endswith("output.mp4") else None),
    )

    def submit(endpoint, arguments, *, skip_cache=False):
        submitted.update(
            endpoint=endpoint, arguments=arguments, skip_cache=skip_cache
        )
        return {"video": {"url": "https://fal.media/output.mp4"}}

    monkeypatch.setattr(
        module.ApiHandler, "submit_and_get_result", staticmethod(submit)
    )

    result = node_cls().edit_video(
        object(),
        "Turn the daytime scene into night",
        resolution="1080p",
        generate_audio=False,
        bitrate_mode="high",
        seed=123,
        force_rerun=True,
    )

    assert submitted == {
        "endpoint": "bytedance/seedance-2.5/reference-to-video",
        "arguments": {
            "prompt": "Turn the daytime scene into night",
            "task": "editing",
            "video_urls": ["https://fal.media/input.mp4"],
            "resolution": "1080p",
            "generate_audio": False,
            "bitrate_mode": "high",
            "seed": 123,
        },
        "skip_cache": True,
    }
    assert result == (native_output, "https://fal.media/output.mp4")
    assert math.isnan(node_cls.IS_CHANGED(force_rerun=True))


def test_seedance25_url_backed_video_passes_through_without_upload(
    pack, monkeypatch
):
    node_cls, module = _node_and_module(pack)
    source_url = "https://fal.media/already-uploaded.mp4"
    captured = {}

    def unexpected_upload(_value):
        raise AssertionError("URL-backed VIDEO input must not be re-uploaded")

    monkeypatch.setattr(
        module.ImageUtils, "upload_file", staticmethod(unexpected_upload)
    )
    monkeypatch.setattr(
        module.MediaUtils,
        "video_from_url",
        staticmethod(lambda _url: object()),
    )

    def submit(_endpoint, arguments, *, skip_cache=False):
        captured.update(arguments)
        assert skip_cache is False
        return {"video": {"url": "https://fal.media/output.mp4"}}

    monkeypatch.setattr(
        module.ApiHandler, "submit_and_get_result", staticmethod(submit)
    )

    node_cls().edit_video(_URLVideo(source_url), "Restyle as watercolor")

    assert captured["video_urls"] == [source_url]
    assert captured["task"] == "editing"
    assert "seed" not in captured
