"""Unit tests for result→ComfyUI-outputs mapping (media decode stubbed)."""

from __future__ import annotations

import json

import pytest
from helpers import _model

_VIDEO_SENTINEL = object()
_AUDIO_SENTINEL = {"waveform": "stub", "sample_rate": 44100}


class _FakeMediaUtils:
    @staticmethod
    def video_from_url(_url):
        return _VIDEO_SENTINEL

    @staticmethod
    def audio_from_url(_url):
        return _AUDIO_SENTINEL


@pytest.fixture(autouse=True)
def _stub_media(monkeypatch, outputs_mod):
    monkeypatch.setattr(outputs_mod, "MediaUtils", _FakeMediaUtils)


def test_return_specs_cover_all_kinds(outputs_mod):
    assert set(outputs_mod.RETURN_SPECS) >= {
        "images", "image", "video", "audio", "text", "file", "json",
    }
    for types, names in outputs_mod.RETURN_SPECS.values():
        assert len(types) == len(names)


def test_video_result(outputs_mod):
    model = _model([], output_kind="video", output_props=["video"])
    result = {"video": {"url": "https://fal.media/v.mp4"}}
    out = outputs_mod.process_result(model, result)
    assert out == (_VIDEO_SENTINEL, "https://fal.media/v.mp4")


def test_audio_result(outputs_mod):
    model = _model([], output_kind="audio", output_props=["audio"])
    result = {"audio": {"url": "https://fal.media/a.mp3"}}
    out = outputs_mod.process_result(model, result)
    assert out == (_AUDIO_SENTINEL, "https://fal.media/a.mp3")


def test_text_result(outputs_mod):
    model = _model([], output_kind="text", output_props=["text"])
    assert outputs_mod.process_result(model, {"text": "hello"}) == ("hello",)


def test_file_result_digs_url(outputs_mod):
    model = _model([], output_kind="file", output_props=["model_glb"])
    result = {"model_glb": {"url": "https://fal.media/m.glb"}}
    assert outputs_mod.process_result(model, result) == ("https://fal.media/m.glb",)


def test_json_fallback(outputs_mod):
    model = _model([], output_kind="json", output_props=[])
    result = {"anything": [1, 2, 3]}
    (payload,) = outputs_mod.process_result(model, result)
    assert json.loads(payload) == result


def test_find_url_recursive(outputs_mod):
    nested = {"a": [{"b": {"url": "https://x/y.bin"}}]}
    assert outputs_mod.find_url(nested) == "https://x/y.bin"
    assert outputs_mod.find_url({"no": "url here"}) is None
