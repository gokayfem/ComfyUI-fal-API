from __future__ import annotations

import importlib

from conftest import PKG, _load_package
from helpers import _model


def test_deprecated_endpoint_keeps_key_in_compatibility_tier():
    _load_package()
    loader = importlib.import_module(f"{PKG}.nodes.dynamic.registry_loader")
    model = _model(
        [],
        endpoint_id="fal-ai/retired",
        deprecated=True,
        deprecated_reason="Endpoint retired.",
    )

    classes, display, skipped, flagged, deprecated = loader._build_model_mappings(
        [model], set()
    )

    key = "FalAPI_fal-ai-retired"
    assert key in classes
    assert classes[key].CATEGORY == "FAL/Compatibility/text-to-image"
    assert display[key].startswith("[Unavailable]")
    assert "absent from the latest live fal catalog" in classes[key].DESCRIPTION
    assert (skipped, flagged, deprecated) == (0, 0, 1)


def test_deprecated_endpoint_is_not_marked_as_superseded():
    _load_package()
    loader = importlib.import_module(f"{PKG}.nodes.dynamic.registry_loader")
    models = [
        _model(
            [],
            endpoint_id="fal-ai/old",
            family="family",
            published_at="2025-01-01T00:00:00Z",
            deprecated=True,
        ),
        _model(
            [],
            endpoint_id="fal-ai/new",
            family="family",
            published_at="2026-01-01T00:00:00Z",
        ),
    ]
    assert loader._superseded_map(models) == {}
