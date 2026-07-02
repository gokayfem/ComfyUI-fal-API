"""Unit tests for the /fal_api server routes' pure functions."""

from __future__ import annotations

import importlib

import pytest
from conftest import PKG, _load_package


@pytest.fixture(scope="session")
def routes():
    _load_package()
    return importlib.import_module(f"{PKG}.nodes.server_routes")


def test_import_without_comfy_server_is_safe(routes):
    # loaded via conftest without ComfyUI's `server` module present
    assert routes is not None


def test_pricing_map_covers_registry(routes):
    pricing_map = routes._pricing_map()
    assert len(pricing_map) > 100
    sample = next(iter(pricing_map.values()))
    assert "label" in sample
    for key in pricing_map:
        assert key.startswith("FalAPI_")


def test_search_models(routes):
    results = routes._search_models(q="kling", category="", max_price=None, limit=10)
    assert results
    assert all("kling" in r["endpoint_id"].lower() or "kling" in r["title"].lower() for r in results)


def test_search_models_price_filter(routes):
    unfiltered = routes._search_models(q="", category="", max_price=None, limit=100)
    cheap = routes._search_models(q="", category="", max_price=0.02, limit=100)
    assert len(cheap) < len(unfiltered)


def test_session_shape(routes):
    payload = routes._session()
    assert set(payload) >= {"total_usd", "calls"}


def test_jobs_degrades_gracefully(routes):
    payload = routes._jobs(limit=5)
    assert "jobs" in payload and "counts" in payload
