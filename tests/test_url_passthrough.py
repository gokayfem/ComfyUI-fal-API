"""Unit tests for fal-CDN URL passthrough (twin inputs + URL outputs)."""

from __future__ import annotations

import pytest
from helpers import _input, _model


def test_media_inputs_get_direct_url_twins(schema_to_inputs):
    model = _model([
        _input("image_url", "string", required=True, media_kind="image"),
        _input("video_url", "string", media_kind="video"),
        _input("doc_url", "string", media_kind="file"),
    ])
    it = schema_to_inputs.build_input_types(model)
    assert "image_url_direct_url" in it["optional"]
    assert "video_url_direct_url" in it["optional"]
    assert "doc_url_direct_url" not in it["optional"]  # file kind excluded


def test_direct_url_wins_over_tensor(arguments_mod, monkeypatch):
    calls = []
    monkeypatch.setattr(
        arguments_mod.ImageUtils,
        "upload_image",
        staticmethod(lambda v: calls.append(v) or "https://uploaded/x.png"),
    )
    model = _model([_input("image_url", "string", media_kind="image")])
    args = arguments_mod.build_arguments(
        model,
        {"image_url": object(), "image_url_direct_url": "https://fal.media/direct.png"},
    )
    assert args["image_url"] == "https://fal.media/direct.png"
    assert not calls  # no upload happened


def test_direct_url_works_without_tensor(arguments_mod):
    model = _model([_input("image_url", "string", media_kind="image")])
    args = arguments_mod.build_arguments(
        model, {"image_url_direct_url": "https://fal.media/direct.png"}
    )
    assert args["image_url"] == "https://fal.media/direct.png"


def test_invalid_direct_url_raises(arguments_mod, errors_mod):
    model = _model([_input("image_url", "string", media_kind="image")])
    with pytest.raises(errors_mod.FalApiError):
        arguments_mod.build_arguments(
            model, {"image_url_direct_url": "not-a-url"}
        )


def test_direct_url_list_splits_commas(arguments_mod):
    model = _model([
        _input("image_urls", "array", media_kind="image", is_list=True)
    ])
    args = arguments_mod.build_arguments(
        model,
        {"image_urls_direct_url": "https://a/1.png, https://b/2.png"},
    )
    assert args["image_urls"] == ["https://a/1.png", "https://b/2.png"]


def test_images_output_includes_urls(outputs_mod):
    types, names = outputs_mod.RETURN_SPECS["images"]
    assert types == ("IMAGE", "STRING")
    assert names == ("images", "image_urls")
