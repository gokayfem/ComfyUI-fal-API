"""Unit tests for provenance sidecars and reproduce-from-file."""

from __future__ import annotations

import importlib
import json

import pytest
from conftest import PKG, _load_package


@pytest.fixture()
def platform():
    _load_package()
    return importlib.import_module(f"{PKG}.nodes.platform_node")


def test_find_request_by_url_escapes_wildcards(platform):
    cache_mod = importlib.import_module(f"{PKG}.nodes.utils.result_cache")
    cache = cache_mod.ResultCache()
    cache.clear()
    url = "https://fal.media/files/a%20b/out_1.png"
    cache.put(
        "fal-ai/flux-2",
        {"prompt": "x"},
        {"images": [{"url": url}]},
        "req-prov",
    )
    hit = cache.find_request_by_url(url)
    assert hit == {"endpoint_id": "fal-ai/flux-2", "request_id": "req-prov"}
    # a percent sign must not act as a wildcard
    assert cache.find_request_by_url("https://fal.media/files/aXb/out_1.png") is None
    cache.clear()


def test_provenance_from_sidecar(platform, tmp_path):
    saved = tmp_path / "out_00001.mp4"
    saved.write_bytes(b"fake video")
    sidecar = tmp_path / "out_00001.mp4.fal.json"
    sidecar.write_text(json.dumps({
        "version": 1,
        "endpoint_id": "fal-ai/veo3",
        "request_id": "req-42",
        "source_url": "https://fal.media/v.mp4",
        "saved_at": 0,
    }))
    node = platform.FalProvenanceFromFile()
    endpoint, request_id, blob = node.read(file_path=str(saved))
    assert endpoint == "fal-ai/veo3"
    assert request_id == "req-42"
    assert json.loads(blob)["source_url"] == "https://fal.media/v.mp4"


def test_provenance_missing_raises(platform, tmp_path, errors_mod):
    bare = tmp_path / "no_provenance.bin"
    bare.write_bytes(b"data")
    node = platform.FalProvenanceFromFile()
    with pytest.raises(errors_mod.FalApiError):
        node.read(file_path=str(bare))


def test_png_chunk_roundtrip(platform, tmp_path):
    from PIL import Image

    png = tmp_path / "img.png"
    Image.new("RGB", (4, 4), "red").save(png)
    payload = {"version": 1, "endpoint_id": "fal-ai/flux-2", "request_id": "req-png"}
    platform._embed_png_provenance(str(png), payload)
    node = platform.FalProvenanceFromFile()
    endpoint, request_id, _ = node.read(file_path=str(png))
    assert endpoint == "fal-ai/flux-2"
    assert request_id == "req-png"
