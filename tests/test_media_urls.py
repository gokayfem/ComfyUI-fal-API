"""Regression tests for malformed media URLs and legacy Seedance output."""

from __future__ import annotations

import sys

import pytest


def test_download_rejects_missing_schema_before_requests(monkeypatch, media_mod):
    called = False

    def unexpected_get(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("requests.get must not receive a malformed URL")

    monkeypatch.setattr(media_mod.requests, "get", unexpected_get)

    with pytest.raises(media_mod.FalApiError, match=r"Expected an HTTP\(S\) media URL"):
        media_mod.MediaUtils.download_url_to_temp("E", ".mp4")

    assert called is False


def test_url_validator_normalizes_whitespace(media_mod):
    assert (
        media_mod.MediaUtils.require_http_url("  https://fal.media/video.mp4  ")
        == "https://fal.media/video.mp4"
    )


def test_seedance_pro_rejects_non_url_result(pack, monkeypatch):
    node_cls = pack.NODE_CLASS_MAPPINGS["SeedanceProImageToVideo_fal"]
    module = sys.modules[node_cls.__module__]

    monkeypatch.setattr(
        module.ImageUtils,
        "upload_image",
        staticmethod(lambda _image: "https://fal.media/input.png"),
    )
    monkeypatch.setattr(
        module.ApiHandler,
        "submit_multiple_and_get_results",
        staticmethod(lambda *_args, **_kwargs: [{"video": {"url": "E"}}]),
    )

    with pytest.raises(module.FalApiError, match=r"Expected an HTTP\(S\) media URL"):
        node_cls().generate_video("prompt", object(), "5")


def test_seedance_pro_returns_validated_url_list(pack, monkeypatch):
    node_cls = pack.NODE_CLASS_MAPPINGS["SeedanceProImageToVideo_fal"]
    module = sys.modules[node_cls.__module__]

    monkeypatch.setattr(
        module.ImageUtils,
        "upload_image",
        staticmethod(lambda _image: "https://fal.media/input.png"),
    )
    monkeypatch.setattr(
        module.ApiHandler,
        "submit_multiple_and_get_results",
        staticmethod(
            lambda *_args, **_kwargs: [
                {"video": {"url": "https://fal.media/output.mp4"}}
            ]
        ),
    )

    assert node_cls().generate_video("prompt", object(), "5") == (
        ["https://fal.media/output.mp4"],
    )
