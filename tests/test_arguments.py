"""Unit tests for kwargs→API-arguments translation (uploads stubbed)."""

from __future__ import annotations

import pytest
from helpers import _input, _model


class _FakeImageUtils:
    @staticmethod
    def upload_image(_value):
        return "https://fal.media/img.png"

    @staticmethod
    def prepare_images(_value):
        return ["https://fal.media/img1.png", "https://fal.media/img2.png"]


class _FakeMediaUtils:
    @staticmethod
    def upload_video(_value):
        return "https://fal.media/vid.mp4"

    @staticmethod
    def upload_audio(_value):
        return "https://fal.media/aud.wav"


@pytest.fixture(autouse=True)
def _stub_uploads(monkeypatch, arguments_mod):
    monkeypatch.setattr(arguments_mod, "ImageUtils", _FakeImageUtils)
    monkeypatch.setattr(arguments_mod, "MediaUtils", _FakeMediaUtils)


def test_seed_minus_one_omitted(arguments_mod):
    model = _model([_input("seed", "integer")])
    args = arguments_mod.build_arguments(model, {"seed": -1})
    assert "seed" not in args


def test_seed_value_sent(arguments_mod):
    model = _model([_input("seed", "integer")])
    args = arguments_mod.build_arguments(model, {"seed": 42})
    assert args["seed"] == 42


def test_custom_size_expands_to_object(arguments_mod):
    model = _model([
        _input("image_size", "enum", enum=["square", "custom_size"],
               default="square", has_custom_size=True)
    ])
    args = arguments_mod.build_arguments(
        model, {"image_size": "custom_size", "width": 832, "height": 1216}
    )
    assert args["image_size"] == {"width": 832, "height": 1216}


def test_preset_size_passes_through(arguments_mod):
    model = _model([
        _input("image_size", "enum", enum=["square", "custom_size"],
               default="square", has_custom_size=True)
    ])
    args = arguments_mod.build_arguments(
        model, {"image_size": "square", "width": 832, "height": 1216}
    )
    assert args["image_size"] == "square"
    assert "width" not in args and "height" not in args


def test_image_upload_single_and_list(arguments_mod):
    model = _model([
        _input("image_url", "string", media_kind="image"),
        _input("image_urls", "array", media_kind="image", is_list=True),
    ])
    args = arguments_mod.build_arguments(
        model, {"image_url": object(), "image_urls": object()}
    )
    assert args["image_url"] == "https://fal.media/img.png"
    assert args["image_urls"] == [
        "https://fal.media/img1.png",
        "https://fal.media/img2.png",
    ]


def test_video_and_audio_upload(arguments_mod):
    model = _model([
        _input("video_url", "string", media_kind="video"),
        _input("audio_url", "string", media_kind="audio"),
    ])
    args = arguments_mod.build_arguments(
        model, {"video_url": object(), "audio_url": object()}
    )
    assert args["video_url"] == "https://fal.media/vid.mp4"
    assert args["audio_url"] == "https://fal.media/aud.wav"


def test_invalid_json_raises_fal_error(arguments_mod, errors_mod):
    model = _model([_input("loras", "json")])
    with pytest.raises(errors_mod.FalApiError):
        arguments_mod.build_arguments(model, {"loras": "{not json"})


def test_valid_json_parsed(arguments_mod):
    model = _model([_input("loras", "json")])
    args = arguments_mod.build_arguments(model, {"loras": '[{"path": "x"}]'})
    assert args["loras"] == [{"path": "x"}]


def test_empty_optional_string_skipped(arguments_mod):
    model = _model([_input("negative_prompt", "string")])
    args = arguments_mod.build_arguments(model, {"negative_prompt": ""})
    assert "negative_prompt" not in args


def test_multi_enum_split_and_validated(arguments_mod, errors_mod):
    model = _model([
        _input("stems", "enum", enum=["vocals", "drums", "bass"], is_list=True)
    ])
    args = arguments_mod.build_arguments(model, {"stems": "vocals, bass"})
    assert args["stems"] == ["vocals", "bass"]

    assert "stems" not in arguments_mod.build_arguments(model, {"stems": "  "})

    with pytest.raises(errors_mod.FalApiError):
        arguments_mod.build_arguments(model, {"stems": "vocals, kazoo"})


def test_kwargs_not_mutated(arguments_mod):
    model = _model([_input("seed", "integer"), _input("prompt", "string", required=True)])
    kwargs = {"seed": -1, "prompt": "hi"}
    snapshot = dict(kwargs)
    arguments_mod.build_arguments(model, kwargs)
    assert kwargs == snapshot
