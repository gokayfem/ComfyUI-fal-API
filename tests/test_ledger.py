"""Unit tests for the session cost ledger."""

from __future__ import annotations

import importlib
import threading

import pytest
from conftest import PKG, _load_package


@pytest.fixture()
def ledger(pricing_mod=None):
    _load_package()
    mod = importlib.import_module(f"{PKG}.nodes.utils.ledger")
    instance = mod.SessionLedger()
    instance.reset()
    yield instance
    instance.reset()


def test_record_and_totals(ledger):
    ledger.record("fal-ai/a", "req-1", 2.5, 0.10)
    ledger.record("fal-ai/b", "req-2", 1.0, None)
    entries = ledger.entries()
    assert len(entries) == 2
    assert ledger.total_cost() == pytest.approx(0.10)
    assert ledger.unknown_cost_count() == 1


def test_report_mentions_calls(ledger):
    ledger.record("fal-ai/kling-video/v3/pro/image-to-video", "abc123", 12.4, 0.35)
    report = ledger.report()
    assert "kling" in report
    assert "abc123" in report


def test_reset(ledger):
    ledger.record("fal-ai/a", None, 1.0, 0.5)
    ledger.reset()
    assert ledger.entries() == []
    assert ledger.total_cost() == 0.0


def test_thread_safety(ledger):
    def worker():
        for _ in range(100):
            ledger.record("fal-ai/t", None, 0.1, 0.01)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.entries()) == 1000
    assert ledger.total_cost() == pytest.approx(10.0)
