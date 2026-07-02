"""Unit tests for the persistent result + upload cache."""

from __future__ import annotations

import importlib

import pytest
from conftest import PKG, _load_package


@pytest.fixture()
def cache():
    _load_package()
    mod = importlib.import_module(f"{PKG}.nodes.utils.result_cache")
    instance = mod.ResultCache()
    instance.clear()
    yield instance
    instance.clear()


def test_round_trip(cache):
    args = {"prompt": "a cat", "seed": 7}
    assert cache.get("fal-ai/test", args) is None
    cache.put("fal-ai/test", args, {"images": [{"url": "https://x/y.png"}]}, "req-1")
    hit = cache.get("fal-ai/test", args)
    assert hit == {"images": [{"url": "https://x/y.png"}]}


def test_key_is_argument_order_independent(cache):
    a = cache.make_key("fal-ai/test", {"a": 1, "b": 2})
    b = cache.make_key("fal-ai/test", {"b": 2, "a": 1})
    assert a == b
    assert a != cache.make_key("fal-ai/other", {"a": 1, "b": 2})


def test_different_args_miss(cache):
    cache.put("fal-ai/test", {"prompt": "a"}, {"ok": 1})
    assert cache.get("fal-ai/test", {"prompt": "b"}) is None


def test_upload_cache_round_trip(cache):
    assert cache.get_upload("hash123") is None
    cache.put_upload("hash123", "https://fal.media/up.png")
    assert cache.get_upload("hash123") == "https://fal.media/up.png"


def test_clear_and_stats(cache):
    cache.put("fal-ai/test", {"p": 1}, {"ok": 1})
    cache.clear()
    assert cache.get("fal-ai/test", {"p": 1}) is None
    stats = cache.stats()
    assert "entries" in stats and "db_path" in stats
