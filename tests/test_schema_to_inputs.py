"""Unit tests for the schema→INPUT_TYPES converter."""

from __future__ import annotations

from helpers import _input, _model


def test_required_and_optional_buckets(schema_to_inputs):
    model = _model([
        _input("prompt", "string", required=True, multiline=True),
        _input("guidance", "number", default=3.5, min=1, max=20),
    ])
    it = schema_to_inputs.build_input_types(model)
    assert "prompt" in it["required"]
    assert "guidance" in it["optional"]
    assert it["required"]["prompt"][0] == "STRING"
    assert it["required"]["prompt"][1]["multiline"] is True


def test_enum_becomes_dropdown(schema_to_inputs):
    model = _model([_input("style", "enum", enum=["a", "b"], default="b")])
    it = schema_to_inputs.build_input_types(model)
    spec = it["optional"]["style"]
    assert spec[0] == ["a", "b"]
    assert spec[1]["default"] == "b"


def test_int_range_and_default_clamp(schema_to_inputs):
    model = _model([_input("steps", "integer", default=28, min=1, max=50)])
    it = schema_to_inputs.build_input_types(model)
    typ, opts = it["optional"]["steps"]
    assert typ == "INT"
    assert opts["min"] == 1 and opts["max"] == 50 and opts["default"] == 28


def test_seed_spec(schema_to_inputs):
    model = _model([_input("seed", "integer", required=True)])
    it = schema_to_inputs.build_input_types(model)
    # seed is always optional regardless of the API marking it required
    typ, opts = it["optional"]["seed"]
    assert typ == "INT"
    assert opts["default"] == -1
    assert opts["min"] == -1
    assert opts.get("control_after_generate") is True


def test_media_inputs(schema_to_inputs):
    model = _model([
        _input("image_url", "string", required=True, media_kind="image"),
        _input("video_url", "string", media_kind="video"),
        _input("audio_url", "string", media_kind="audio"),
    ])
    it = schema_to_inputs.build_input_types(model)
    assert it["required"]["image_url"][0] == "IMAGE"
    assert it["optional"]["video_url"][0] == "VIDEO"
    assert it["optional"]["audio_url"][0] == "AUDIO"


def test_custom_size_companions(schema_to_inputs):
    model = _model([
        _input(
            "image_size", "enum",
            enum=["square", "landscape_4_3", "custom_size"],
            default="landscape_4_3", has_custom_size=True,
        )
    ])
    it = schema_to_inputs.build_input_types(model)
    assert "width" in it["optional"] and "height" in it["optional"]
    assert it["optional"]["width"][0] == "INT"


def test_dict_default_maps_to_custom_size(schema_to_inputs):
    model = _model([
        _input(
            "image_size", "enum",
            enum=["square", "custom_size"],
            default={"width": 2048, "height": 1536},
            has_custom_size=True,
        )
    ])
    it = schema_to_inputs.build_input_types(model)
    assert it["optional"]["image_size"][1]["default"] == "custom_size"
    assert it["optional"]["width"][1]["default"] == 2048
    assert it["optional"]["height"][1]["default"] == 1536


def test_multi_select_enum_is_comma_string(schema_to_inputs):
    model = _model([
        _input("stems", "enum", enum=["vocals", "drums", "bass"],
               default=["vocals", "drums"], is_list=True)
    ])
    it = schema_to_inputs.build_input_types(model)
    typ, opts = it["optional"]["stems"]
    assert typ == "STRING"
    assert opts["default"] == "vocals, drums"
    assert "vocals, drums, bass" in opts["tooltip"]


def test_json_field_is_multiline_string(schema_to_inputs):
    model = _model([_input("loras", "json")])
    it = schema_to_inputs.build_input_types(model)
    typ, opts = it["optional"]["loras"]
    assert typ == "STRING"
    assert opts["multiline"] is True


def test_force_rerun_always_present(schema_to_inputs):
    it = schema_to_inputs.build_input_types(_model([]))
    typ, opts = it["optional"]["force_rerun"]
    assert typ == "BOOLEAN"
    assert opts["default"] is False


def test_every_input_has_tooltip_when_description_given(schema_to_inputs):
    model = _model([_input("prompt", "string", required=True, description="What to draw")])
    it = schema_to_inputs.build_input_types(model)
    assert it["required"]["prompt"][1]["tooltip"] == "What to draw"
