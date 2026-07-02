"""Validate the committed model registry — pure JSON, no heavy imports."""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "fal_registry.json"

VALID_INPUT_TYPES = {"string", "integer", "number", "boolean", "enum", "object", "array", "json"}
VALID_OUTPUT_KINDS = {"images", "image", "video", "audio", "text", "file", "json"}
VALID_MEDIA_KINDS = {None, "image", "video", "audio", "file"}


def _registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_top_level_shape():
    reg = _registry()
    assert reg["version"] == 1
    assert reg["model_count"] == len(reg["models"])
    assert reg["model_count"] > 500


def test_models_well_formed():
    reg = _registry()
    seen_ids = set()
    for model in reg["models"]:
        eid = model["endpoint_id"]
        assert eid and "/" in eid, f"bad endpoint_id: {eid!r}"
        assert eid not in seen_ids, f"duplicate endpoint_id: {eid}"
        seen_ids.add(eid)
        assert model["title"], f"{eid}: missing title"
        assert model["category"], f"{eid}: missing category"
        assert model["output_kind"] in VALID_OUTPUT_KINDS, f"{eid}: {model['output_kind']}"
        assert isinstance(model["inputs"], list)


def test_inputs_well_formed():
    reg = _registry()
    for model in reg["models"]:
        eid = model["endpoint_id"]
        names = set()
        for inp in model["inputs"]:
            name = inp["name"]
            assert name not in names, f"{eid}: duplicate input {name}"
            names.add(name)
            assert inp["type"] in VALID_INPUT_TYPES, f"{eid}.{name}: {inp['type']}"
            assert inp.get("media_kind") in VALID_MEDIA_KINDS, f"{eid}.{name}"
            if inp["type"] == "enum":
                assert inp.get("enum"), f"{eid}.{name}: enum without values"


def test_enum_defaults_are_members_or_custom_size():
    reg = _registry()
    for model in reg["models"]:
        for inp in model["inputs"]:
            if inp["type"] == "enum" and inp.get("default") is not None:
                if inp["default"] in inp["enum"]:
                    continue
                # two legitimate non-member shapes exist in the wild:
                # 1. has_custom_size enums defaulting to an explicit
                #    {width, height} object (mapped to the custom_size preset)
                # 2. multi-select enums (is_list) defaulting to a list of
                #    members (mapped to a comma-separated string widget)
                if inp.get("has_custom_size") and isinstance(inp["default"], dict):
                    continue
                if inp.get("is_list") and isinstance(inp["default"], list):
                    assert all(v in inp["enum"] for v in inp["default"]), (
                        f"{model['endpoint_id']}.{inp['name']}: list default "
                        f"contains non-members"
                    )
                    continue
                raise AssertionError(
                    f"{model['endpoint_id']}.{inp['name']}: default "
                    f"{inp['default']!r} not in enum"
                )
