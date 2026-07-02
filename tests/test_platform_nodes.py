"""Integration tests for the FAL/Platform utility nodes."""

from __future__ import annotations

import pytest

PLATFORM_KEYS = [
    "FalSubmit_fal",
    "FalCollect_fal",
    "FalResultByRequestId_fal",
    "FalCostEstimator_fal",
    "FalSessionCosts_fal",
    "FalSaveMediaURL_fal",
]


def test_all_platform_nodes_registered(pack):
    for key in PLATFORM_KEYS:
        assert key in pack.NODE_CLASS_MAPPINGS, key
        assert key in pack.NODE_DISPLAY_NAME_MAPPINGS, key
        cls = pack.NODE_CLASS_MAPPINGS[key]
        assert cls.CATEGORY == "FAL/Platform", key
        input_types = cls.INPUT_TYPES()
        assert "required" in input_types or "optional" in input_types


def test_submit_returns_handle_type(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalSubmit_fal"]
    assert "FAL_HANDLE" in cls.RETURN_TYPES


def test_collect_rejects_bad_handle(pack, errors_mod):
    cls = pack.NODE_CLASS_MAPPINGS["FalCollect_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    with pytest.raises(errors_mod.FalApiError):
        fn(handle="not a handle")


def test_cost_estimator_never_raises(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalCostEstimator_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    report, total = fn(endpoint_id="fal-ai/definitely-not-real", runs=5)
    assert isinstance(report, str)
    assert isinstance(total, float)


def test_session_costs_reports(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalSessionCosts_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    out = fn(reset=False)
    report, total = out[0], out[1]
    assert isinstance(report, str)
    assert isinstance(total, float)


def test_save_media_rejects_empty_url(pack, errors_mod):
    cls = pack.NODE_CLASS_MAPPINGS["FalSaveMediaURL_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    with pytest.raises((errors_mod.FalApiError, ValueError)):
        fn(url="", filename_prefix="fal/test")
