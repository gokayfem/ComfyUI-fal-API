"""Unit tests for the durable job inbox store."""

from __future__ import annotations

import importlib

import pytest
from conftest import PKG, _load_package


@pytest.fixture()
def store():
    _load_package()
    mod = importlib.import_module(f"{PKG}.nodes.utils.job_store")
    instance = mod.JobStore()
    instance.prune(older_than_days=0)  # clear anything from other tests
    yield instance
    instance.prune(older_than_days=0)


def test_submit_and_collect_lifecycle(store):
    store.record_submit("fal-ai/kling-video/v3/pro/image-to-video", "req-a")
    store.record_submit("fal-ai/flux-2", "req-b")
    assert store.counts()["submitted"] == 2

    store.mark_collected("req-a")
    counts = store.counts()
    assert counts["submitted"] == 1
    assert counts["collected"] == 1

    pending = store.pending()
    assert len(pending) == 1
    assert pending[0]["request_id"] == "req-b"


def test_entries_newest_first(store):
    store.record_submit("fal-ai/a", "req-1")
    store.record_submit("fal-ai/b", "req-2")
    entries = store.entries()
    assert entries[0]["request_id"] == "req-2"


def test_mark_collected_unknown_id_is_silent(store):
    store.mark_collected("req-from-another-session")
    entries = store.entries(status="collected")
    assert any(e["request_id"] == "req-from-another-session" for e in entries)


def test_report_mentions_pending(store):
    store.record_submit("fal-ai/kling-video/v3/pro/image-to-video", "req-x")
    report = store.report()
    assert "req-x" in report
    assert "pending" in report.lower()


def test_inbox_node_outputs(pack, store):
    store.record_submit("fal-ai/veo3", "req-latest")
    cls = pack.NODE_CLASS_MAPPINGS["FalJobInbox_fal"]
    node = cls()
    out = getattr(node, cls.FUNCTION)(status_filter="all", limit=20)
    report, latest_id, latest_endpoint = out[0], out[1], out[2]
    assert isinstance(report, str)
    assert latest_id == "req-latest"
    assert latest_endpoint == "fal-ai/veo3"
