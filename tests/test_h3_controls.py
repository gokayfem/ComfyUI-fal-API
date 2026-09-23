"""Check shared controls on shipped H3 generation nodes through the API boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "fal_registry.json"
H3_SHARED_CONTROL_MODELS = [
    model for model in json.loads(REGISTRY.read_text())["models"]
    if model["endpoint_id"].startswith(("minimax/h3/", "minimax/h3-max/", "minimax/h3-max-turbo/"))
    and not any(
        specialty in model["endpoint_id"]
        for specialty in ("/trainer", "/lip-sync/", "/styles/")
    )
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
    assert widgets["duration"][0] == "INT"
    assert widgets["duration"][1]["min"] == 5
    assert widgets["duration"][1]["max"] == 15
    assert widgets["duration"][1]["default"] == 5
    bucket = "required" if "h3-max" in model["endpoint_id"] else "optional"
    assert "prompt_expansion_mode" in inputs[bucket]
    assert widgets["prompt_expansion_mode"][0] == "STRING"
    assert "balanced" in widgets["prompt_expansion_mode"][1]["fal_suggestions"]
    assert "quality" in widgets["prompt_expansion_mode"][1]["fal_suggestions"]
    resolution = "1080P" if "h3-max" in model["endpoint_id"] else "2K"
    assert resolution in widgets["resolution"][0]
    cls().run(prompt="A camera pans", duration=15, resolution=resolution, prompt_expansion_mode="quality")
    assert captured == [(model["endpoint_id"], {
        "prompt": "A camera pans", "duration": 15, "resolution": resolution, "prompt_expansion_mode": "quality",
    }, False)]
