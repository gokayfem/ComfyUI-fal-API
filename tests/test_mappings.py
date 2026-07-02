"""Integration: the loaded pack must keep every legacy key and stay coherent."""

from __future__ import annotations

import json
from pathlib import Path

LEGACY_SNAPSHOT = Path(__file__).with_name("legacy_node_keys.json")


def test_all_legacy_keys_present(pack):
    """Backward-compat lock: keys registered at v1.0.12 must never disappear."""
    legacy = set(json.loads(LEGACY_SNAPSHOT.read_text())["keys"])
    current = set(pack.NODE_CLASS_MAPPINGS)
    missing = legacy - current
    assert not missing, f"legacy node keys removed (breaks user workflows): {sorted(missing)}"


def test_display_names_complete(pack):
    missing = [k for k in pack.NODE_CLASS_MAPPINGS if k not in pack.NODE_DISPLAY_NAME_MAPPINGS]
    assert not missing


def test_dynamic_nodes_registered(pack):
    dynamic = [k for k in pack.NODE_CLASS_MAPPINGS if k.startswith("FalAPI_")]
    assert len(dynamic) > 500, "dynamic registry failed to load"
    assert "FalAnyEndpoint_fal" in pack.NODE_CLASS_MAPPINGS


def test_every_node_class_is_valid(pack):
    for key, cls in pack.NODE_CLASS_MAPPINGS.items():
        input_types = cls.INPUT_TYPES()
        assert isinstance(input_types, dict), key
        assert "required" in input_types or "optional" in input_types, key
        assert isinstance(cls.RETURN_TYPES, tuple), key
        assert isinstance(cls.FUNCTION, str) and hasattr(cls, cls.FUNCTION), key
        assert isinstance(cls.CATEGORY, str) and cls.CATEGORY, key


def test_no_bare_video_category_left(pack):
    bare = [
        k for k, cls in pack.NODE_CLASS_MAPPINGS.items()
        if cls.CATEGORY.lower() == "video"
    ]
    assert not bare, f"nodes escaped the FAL/ menu namespace: {bare}"
