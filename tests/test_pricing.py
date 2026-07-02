"""Unit tests for the registry pricing parser."""

from __future__ import annotations

import importlib

import pytest
from conftest import PKG, _load_package


@pytest.fixture(scope="session")
def pricing_mod():
    _load_package()
    return importlib.import_module(f"{PKG}.nodes.utils.pricing")


def test_run_ratio_wins_over_per_unit(pricing_mod):
    text = (
        "Your request will cost $0.08 per image. For $1.00, you can run "
        "this model 12 times."
    )
    parsed = pricing_mod.PricingUtils.parse(text)
    assert parsed["per_run"] == pytest.approx(1.0 / 12.0)


def test_per_unit_only(pricing_mod):
    parsed = pricing_mod.PricingUtils.parse(
        "Your request will cost $0.05 per second of video."
    )
    assert parsed["per_run"] is None
    assert parsed["per_unit"] == pytest.approx(0.05)
    assert "second" in parsed["unit"]


def test_per_image_implies_per_run(pricing_mod):
    parsed = pricing_mod.PricingUtils.parse("Your request will cost $0.04 per image.")
    assert parsed["per_run"] == pytest.approx(0.04)


def test_junk_never_raises(pricing_mod):
    for junk in ("", "free during preview!!", "$", "per per per", None or ""):
        parsed = pricing_mod.PricingUtils.parse(junk)
        assert parsed["raw"] == junk


def test_estimate_against_real_registry(pricing_mod):
    est = pricing_mod.PricingUtils.estimate("fal-ai/nano-banana-2/edit", 10)
    assert est["runs"] == 10
    report = pricing_mod.PricingUtils.format_report(est)
    assert "fal-ai/nano-banana-2/edit" in report


def test_unknown_endpoint_safe(pricing_mod):
    est = pricing_mod.PricingUtils.estimate("fal-ai/does-not-exist", 3)
    report = pricing_mod.PricingUtils.format_report(est)
    assert isinstance(report, str) and report
