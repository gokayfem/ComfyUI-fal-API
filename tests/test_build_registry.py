"""Offline regressions for controls lost or mistyped during schema distillation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.build_registry import (
    build_record,
    distill_inputs,
    distill_property,
    normalize_schema,
)

H3_INPUTS = json.loads((Path(__file__).parent / "fixtures" / "h3_inputs.json").read_text())


def _doc(schema):
    return {"components": {"schemas": {"ModelInput": schema}}}


@pytest.mark.parametrize("endpoint_id", H3_INPUTS)
def test_h3_controls_from_upstream_schema(endpoint_id):
    inputs = distill_inputs(H3_INPUTS[endpoint_id]["input"], {}, endpoint_id)
    by_name = {inp["name"]: inp for inp in inputs}
    assert by_name["duration"]["type"] == "integer"
    assert (by_name["duration"]["min"], by_name["duration"]["max"]) == (5, 15)
    assert by_name["duration"]["default"] == 5
    assert by_name["prompt_expansion_mode"]["type"] == "string"
    assert by_name["prompt_expansion_mode"]["default"] == "balanced"
    required = H3_INPUTS[endpoint_id]["input"]["required"]
    assert by_name["prompt_expansion_mode"]["required"] == ("prompt_expansion_mode" in required)
    if "h3-max" in endpoint_id:
        assert by_name["resolution"]["enum"] == ["480P", "768P", "1080P"]
        assert by_name["prompt_expansion_mode"]["suggestions"] == ["disabled", "balanced", "quality"]
    else:
        assert by_name["resolution"]["enum"] == ["480P", "768P", "2K", "4K"]
        assert by_name["prompt_expansion_mode"]["suggestions"] == ["disabled", "fast", "balanced", "quality"]


@pytest.mark.parametrize("name,examples", [
    ("mode", ["balanced", "quality"]),
    ("language", ["en", "tr", "ja"]),
    ("voice", ["Aria", "Rachel"]),
    ("model", ["vendor/model-a", "vendor/model-b"]),
])
def test_suggestions_are_generic_and_preserve_free_text(name, examples):
    raw = {"type": "string", "examples": examples, "default": "new-value"}
    original = copy.deepcopy(raw)
    inp = distill_property(name, raw, set(), {})
    assert inp["type"] == "string"
    assert inp["enum"] is None
    assert inp["suggestions"] == examples
    assert inp["default"] == "new-value"
    assert raw == original


@pytest.mark.parametrize("name,raw", [
    ("prompt", {"type": "string", "examples": ["cat", "dog"]}),
    ("custom_prompt", {"type": "string", "examples": ["cat", "dog"]}),
    ("description", {"type": "string", "examples": ["a cat", "a dog"]}),
    ("image_url", {"type": "string", "examples": ["https://example.com/a", "https://example.com/b"]}),
    ("contact", {"type": "string", "format": "email", "examples": ["a", "b"]}),
    ("mode", {"type": "string", "examples": ["only-one"]}),
    ("mode", {"type": "string", "enum": ["a", "b"], "examples": ["a", "c"]}),
])
def test_free_text_and_true_enums_do_not_get_suggestions(name, raw):
    assert "suggestions" not in distill_property(name, raw, set(), {})


def test_large_schema_does_not_silently_drop_optional_controls():
    properties = {f"field_{i}": {"type": "boolean"} for i in range(50)}
    properties["duration"] = {"type": "integer", "minimum": 5, "maximum": 15}
    inputs = distill_inputs({"properties": properties}, {}, "any/future-model")
    assert len(inputs) == 51
    assert inputs[-1]["name"] == "duration"


def test_nested_refs_and_nullable_composition_keep_controls():
    components = {
        "Alias": {"$ref": "#/components/schemas/Resolution"},
        "Resolution": {"allOf": [{"type": "string", "enum": ["768P", "1080P"]}]},
    }
    inp = distill_property("resolution", {
        "anyOf": [{"$ref": "#/components/schemas/Alias"}, {"type": "null"}],
        "default": "1080P",
        "description": "Output resolution",
    }, set(), components)
    assert inp["type"] == "enum"
    assert inp["enum"] == ["768P", "1080P"]
    assert inp["default"] == "1080P"
    assert inp["description"] == "Output resolution"


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_union_of_literals_preserves_all_choices(keyword):
    inp = distill_property("resolution", {keyword: [
        {"const": "480P"}, {"enum": ["768P", "1080P"]}, {"type": "null"},
    ]}, set(), {})
    assert inp["enum"] == ["480P", "768P", "1080P"]


def test_literal_union_survives_nested_composition_and_repeated_normalization():
    raw = {"allOf": [{"anyOf": [{"const": "480P"}, {"const": "1080P"}]}]}
    normalized = normalize_schema(raw, {})[0]
    assert normalized["enum"] == ["480P", "1080P"]
    assert normalize_schema(normalized, {})[0] == normalized


@pytest.mark.parametrize("schema", [
    {"allOf": [{"enum": ["480P", "768P"]}, {"enum": ["768P", "1080P"]}]},
    {"anyOf": [{"const": "480P"}, {"const": "768P"}], "enum": ["768P"]},
])
def test_composed_enum_respects_intersecting_constraints(schema):
    assert distill_property("resolution", schema, set(), {})["enum"] == ["768P"]


def test_all_of_input_objects_keep_inherited_fields_and_required_names():
    doc = _doc({"allOf": [
        {"$ref": "#/components/schemas/Base"},
        {"properties": {"image_url": {"type": "string"}}, "required": ["image_url"]},
    ], "properties": {"duration": {"type": "integer", "default": 5, "minimum": 5, "maximum": 15}}})
    doc["components"]["schemas"]["Base"] = {
        "properties": {"prompt": {"type": "string"}}, "required": ["prompt"],
    }
    record = build_record({"id": "test/model"}, doc)
    assert [inp["name"] for inp in record["inputs"]] == ["prompt", "image_url", "duration"]
    assert [inp["required"] for inp in record["inputs"]] == [True, True, False]


def test_nullable_type_array_is_a_numeric_control():
    inp = distill_property("duration", {"type": ["integer", "null"], "default": 5}, set(), {})
    assert inp["type"] == "integer"


def test_reference_cycles_do_not_recurse_forever():
    components = {"A": {"$ref": "#/components/schemas/B"}, "B": {"$ref": "#/components/schemas/A"}}
    assert normalize_schema({"$ref": "#/components/schemas/A"}, components) == ({}, False, None)


def test_custom_image_size_still_has_preset_and_dimensions():
    inp = distill_property("image_size", {"anyOf": [
        {"type": "string", "enum": ["square", "landscape"]},
        {"type": "object", "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}}},
    ]}, set(), {})
    assert inp["has_custom_size"] is True
    assert inp["enum"] == ["square", "landscape", "custom_size"]


def test_nullable_nested_custom_size_preserves_dimension_controls():
    inp = distill_property("image_size", {"anyOf": [
        {"allOf": [{"anyOf": [
            {"enum": ["square", "landscape"]},
            {"type": "object", "properties": {"width": {}, "height": {}}},
        ]}]},
        {"const": None},
    ]}, set(), {})
    assert inp["has_custom_size"] is True
    assert inp["enum"] == ["square", "landscape", "custom_size"]


def test_open_string_union_keeps_literal_suggestions_without_restricting_values():
    inp = distill_property("voice", {"anyOf": [
        {"enum": ["Aria", "Rachel"]}, {"type": "string"},
    ]}, set(), {})
    assert inp["type"] == "string"
    assert inp["enum"] is None
    assert inp["suggestions"] == ["Aria", "Rachel"]


def test_exact_endpoint_path_wins_over_first_path_and_accepts_inline_schema():
    def request(schema):
        return {"post": {"requestBody": {"content": {"application/json": {"schema": schema}}}}}

    doc = _doc({"properties": {"wrong_field": {"type": "string"}}})
    doc["paths"] = {
        "/test/other": request({"$ref": "#/components/schemas/ModelInput"}),
        "/test/model": request({"properties": {"duration": {"type": "integer", "default": 5}}}),
    }
    record = build_record({"id": "test/model"}, doc)
    assert [inp["name"] for inp in record["inputs"]] == ["duration"]
