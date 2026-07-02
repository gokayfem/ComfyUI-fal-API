"""Unit tests for the spend guard and balance node."""

from __future__ import annotations

import importlib

import pytest
from conftest import PKG, _load_package


@pytest.fixture()
def billing_mod():
    _load_package()
    return importlib.import_module(f"{PKG}.nodes.utils.billing")


def test_preflight_noop_when_unconfigured(billing_mod):
    # no [spend_guard] section in config → both checks disabled
    billing_mod.SpendGuard.preflight("fal-ai/anything")


def test_balance_node_never_raises(pack, monkeypatch, billing_mod):
    monkeypatch.setattr(
        billing_mod.BillingUtils, "get_balance", staticmethod(lambda force=False: None)
    )
    cls = pack.NODE_CLASS_MAPPINGS["FalBalance_fal"]
    node = cls()
    out = getattr(node, cls.FUNCTION)(force_refresh=False)
    report, balance = out[0], out[1]
    assert isinstance(report, str) and report
    assert balance == -1.0


def test_balance_node_reports_value(pack, monkeypatch, billing_mod):
    monkeypatch.setattr(
        billing_mod.BillingUtils, "get_balance", staticmethod(lambda force=False: 24.5)
    )
    cls = pack.NODE_CLASS_MAPPINGS["FalBalance_fal"]
    node = cls()
    out = getattr(node, cls.FUNCTION)(force_refresh=True)
    assert out[1] == pytest.approx(24.5)
