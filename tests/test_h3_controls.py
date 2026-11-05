"""Check shared controls on shipped H3 generation nodes through the API boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "fal_registry.json"
# Only generation operations share these controls. New editing/extension
# endpoints must not silently inherit generation-specific assertions.
H3_GENERATION_OPERATIONS = {
    "image-to-video", "text-to-video", "reference-to-video", "camera-controls", "multi-angle/image-to-video",
}
H3_SHARED_CONTROL_MODELS = [
    model for model in json.loads(REGISTRY.read_text())["models"]
    if model["endpoint_id"].startswith(("minimax/h3/", "minimax/h3-max/", "minimax/h3-max-turbo/"))
    and model["endpoint_id"].split("/", 2)[2] in H3_GENERATION_OPERATIONS
    and not model.get("deprecated")
]


@pytest.mark.parametrize("model", H3_SHARED_CONTROL_MODELS, ids=lambda model: model["endpoint_id"])
def test_shipped_h3_shared_control_widgets_and_api_arguments(model, factory_mod, monkeypatch):
    monkeypatch.setattr(factory_mod, "_ASYNC_CAPABLE", False)
    captured = []
    monkeypatch.setattr(factory_mod, "_call_api", lambda *args: captured.append(args) or {})
    monkeypatch.setattr(factory_mod, "process_result", lambda *args: ())
    cls = factory_mod.build_node_class(model)
    inputs = cls.INPUT_TYPES()
    widgets = {**inputs["required"], **inputs["optional"]}
    duration = next(inp for inp in model["inputs"] if inp["name"] == "duration")
    assert duration["type"] in {"integer", "number"}
    assert widgets["duration"][0] == ("INT" if duration["type"] == "integer" else "FLOAT")
    assert widgets["duration"][1]["min"] == duration["min"]
    assert widgets["duration"][1]["max"] == duration["max"]
    assert widgets["duration"][1]["default"] == duration["default"]
    bucket = "required" if "h3-max" in model["endpoint_id"] else "optional"
    assert "prompt_expansion_mode" in inputs[bucket]
    assert widgets["prompt_expansion_mode"][0] == "STRING"
    assert "balanced" in widgets["prompt_expansion_mode"][1]["fal_suggestions"]
    assert "quality" in widgets["prompt_expansion_mode"][1]["fal_suggestions"]
    resolution = "1080P" if "h3-max" in model["endpoint_id"] else "2K"
    assert resolution in widgets["resolution"][0]
    seconds = duration["default"]
    cls().run(prompt="A camera pans", duration=seconds, resolution=resolution, prompt_expansion_mode="quality")
    assert captured == [(model["endpoint_id"], {
        "prompt": "A camera pans", "duration": seconds, "resolution": resolution, "prompt_expansion_mode": "quality",
    }, False)]


def test_h3_fractional_duration_reaches_api_without_truncation(factory_mod, monkeypatch):
    model = deepcopy(next(model for model in H3_SHARED_CONTROL_MODELS
                          if model["endpoint_id"] == "minimax/h3-max/image-to-video"))
    duration = next(inp for inp in model["inputs"] if inp["name"] == "duration")
    duration.update(type="number", min=0.92, max=15, default=5)
    monkeypatch.setattr(factory_mod, "_ASYNC_CAPABLE", False)
    captured = []
    monkeypatch.setattr(factory_mod, "_call_api", lambda *args: captured.append(args) or {})
    monkeypatch.setattr(factory_mod, "process_result", lambda *args: ())
    cls = factory_mod.build_node_class(model)
    cls().run(prompt="A camera pans", duration=2.5)
    assert captured[0][1]["duration"] == 2.5
