"""Unit tests for the /fal_api server routes' pure functions."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("failure_step", [None, "build_registry.py", "validate_registry.py"])
def test_refresh_promotes_only_validated_candidates(routes, monkeypatch, tmp_path, failure_step):
    import subprocess

    data = tmp_path / "data"
    data.mkdir()
    baseline = data / "fal_registry.json"
    baseline.write_text("original registry")
    monkeypatch.setattr(routes, "_repo_root", lambda: str(tmp_path))
    steps = []

    def run(command, **kwargs):
        step = Path(command[1]).name
        steps.append(step)
        assert baseline.read_text() == "original registry"
        assert command[-2:] == ["--preserve-from" if step == "build_registry.py" else "--baseline", str(baseline)]
        if step == "build_registry.py":
            Path(command[3]).write_text("validated candidate")
        else:
            assert Path(command[2]).read_text() == "validated candidate"
        return SimpleNamespace(returncode=int(step == failure_step), stdout="", stderr="schema rejected")

    monkeypatch.setattr(subprocess, "run", run)
    ok, message = routes._run_refresh_subprocess()
    assert ok == (failure_step is None)
    assert baseline.read_text() == ("validated candidate" if ok else "original registry")
    assert len(list(data.iterdir())) == 1
    assert steps[0] == "build_registry.py"
    if failure_step != "build_registry.py":
        assert steps[1] == "validate_registry.py"
    assert "Restart ComfyUI" in message if ok else "schema rejected" in message
