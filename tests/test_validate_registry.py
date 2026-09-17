from __future__ import annotations

import copy

import pytest

from scripts.build_readme import render_generated_section
from scripts.build_registry import preserve_missing_records
from scripts.validate_registry import (
    RegistryValidationError,
    compare_model_inputs,
    compare_registries,
    validate_registry,
)


def _model(endpoint_id: str) -> dict:
    return {
        "endpoint_id": endpoint_id,
        "title": endpoint_id,
        "category": "text-to-image",
        "description": "",
        "family": "",
        "lab": "",
        "pricing": "",
        "published_at": "",
        "inputs": [],
        "output_kind": "images",
        "output_props": [],
        "thumbnail": "",
    }


def _registry(*endpoint_ids: str) -> dict:
    models = [_model(endpoint_id) for endpoint_id in endpoint_ids]
    return {
        "version": 1,
        "model_count": len(models),
        "live_model_count": len(models),
        "deprecated_model_count": 0,
        "models": models,
    }


def test_valid_registry_returns_endpoint_ids():
    assert validate_registry(_registry("fal-ai/a", "fal-ai/b"), min_models=2) == {
        "fal-ai/a",
        "fal-ai/b",
    }


@pytest.mark.parametrize(
    "registry, message",
    [
        ({"version": 1, "models": []}, "top-level"),
        (
            {
                "version": 1,
                "model_count": 2,
                "live_model_count": 1,
                "deprecated_model_count": 0,
                "models": [_model("fal-ai/a")],
            },
            "does not match",
        ),
        (_registry("fal-ai/b", "fal-ai/a"), "sorted"),
        (_registry("fal-ai/a", "fal-ai/a"), "Duplicate"),
    ],
)
def test_invalid_registry_is_rejected(registry, message):
    with pytest.raises(RegistryValidationError, match=message):
        validate_registry(registry, min_models=1)


def test_unknown_output_kind_is_rejected():
    registry = _registry("fal-ai/a")
    registry["models"][0]["output_kind"] = "binary"
    with pytest.raises(RegistryValidationError, match="unknown output_kind"):
        validate_registry(registry, min_models=1)


def test_invalid_endpoint_id_is_rejected():
    with pytest.raises(RegistryValidationError, match="invalid endpoint_id"):
        validate_registry(_registry("fal-ai/bad endpoint"), min_models=1)


def test_deprecated_counts_must_match_records():
    registry = _registry("fal-ai/a")
    registry["models"][0]["deprecated"] = True
    with pytest.raises(RegistryValidationError, match="live_model_count"):
        validate_registry(registry, min_models=1)


def test_large_removal_is_blocked_by_default():
    baseline = {f"fal-ai/{index}" for index in range(100)}
    candidate = {f"fal-ai/{index}" for index in range(90)}
    with pytest.raises(RegistryValidationError, match="above the 5.0% limit"):
        compare_registries(baseline, candidate)


def test_large_removal_can_be_explicitly_allowed():
    baseline = {f"fal-ai/{index}" for index in range(100)}
    candidate = {f"fal-ai/{index}" for index in range(90)}
    added, removed = compare_registries(baseline, candidate, allow_large_change=True)
    assert added == set()
    assert len(removed) == 10


def test_large_addition_is_blocked_by_default():
    baseline = {f"fal-ai/{index}" for index in range(100)}
    candidate = baseline | {f"new/{index}" for index in range(30)}
    with pytest.raises(RegistryValidationError, match="above the 25.0% limit"):
        compare_registries(baseline, candidate)


def test_missing_baseline_record_is_preserved_as_deprecated():
    current = [_model("fal-ai/current")]
    baseline = {"models": [_model("fal-ai/current"), _model("fal-ai/retired")]}
    merged = preserve_missing_records(current, baseline)
    by_id = {model["endpoint_id"]: model for model in merged}
    assert set(by_id) == {"fal-ai/current", "fal-ai/retired"}
    assert "deprecated" not in by_id["fal-ai/current"]
    assert by_id["fal-ai/retired"]["deprecated"] is True


def test_generated_catalog_lists_live_models_only():
    live = _model("fal-ai/live")
    retired = {**_model("fal-ai/retired"), "deprecated": True}
    registry = {
        "models": [live, retired],
        "model_count": 2,
        "live_model_count": 1,
        "deprecated_model_count": 1,
    }
    rendered = render_generated_section(registry)
    assert "1 live models" in rendered
    assert "1 compatibility-preserved" in rendered
    assert "fal-ai/live" in rendered
    assert "fal-ai/retired" not in rendered


def _input_registry():
    registry = _registry("fal-ai/video")
    registry["models"][0]["inputs"] = [
        {"name": "duration", "type": "integer", "default": 5, "required": False},
        {"name": "resolution", "type": "enum", "enum": ["768P", "1080P"], "required": False},
    ]
    return registry


@pytest.mark.parametrize("change, message", [
    (lambda inputs: inputs.pop(0), "removed input duration"),
    (lambda inputs: inputs[1].update(type="string"), "lost enum control"),
    (lambda inputs: inputs[1].update(enum=["768P"]), "removed choices"),
    (lambda inputs: inputs[0].update(type="json"), "lost integer control"),
])
def test_refresh_detects_disappearing_controls(change, message):
    baseline = _input_registry()
    candidate = copy.deepcopy(baseline)
    change(candidate["models"][0]["inputs"])
    assert message in compare_model_inputs(baseline, candidate)[0]


def test_refresh_allows_new_inputs_choices_and_string_to_dropdown():
    baseline = _input_registry()
    baseline["models"][0]["inputs"].append({"name": "mode", "type": "string", "required": False})
    candidate = copy.deepcopy(baseline)
    inputs = candidate["models"][0]["inputs"]
    inputs.append({"name": "seed", "type": "integer", "required": False})
    inputs[1]["enum"].append("4K")
    inputs[2].update(type="enum", enum=["balanced", "quality"])
    assert compare_model_inputs(baseline, candidate) == []


@pytest.mark.parametrize("change, message", [
    (lambda inputs: inputs.append(dict(inputs[0])), "duplicate input"),
    (lambda inputs: inputs[0].update(type="unknown"), "invalid input type"),
    (lambda inputs: inputs[1].update(enum=[]), "no enum choices"),
    (lambda inputs: inputs[0].update(required="false"), "required must be a boolean"),
    (lambda inputs: inputs.append(None), "invalid input record"),
])
def test_refresh_rejects_malformed_controls(change, message):
    registry = _input_registry()
    change(registry["models"][0]["inputs"])
    with pytest.raises(RegistryValidationError, match=message):
        validate_registry(registry, min_models=1)


def test_refresh_rejects_lost_suggested_choices():
    baseline = _input_registry()
    baseline["models"][0]["inputs"].append({
        "name": "mode", "type": "string", "required": False, "suggestions": ["fast", "quality"],
    })
    candidate = copy.deepcopy(baseline)
    del candidate["models"][0]["inputs"][-1]["suggestions"]
    assert "lost suggested choices" in compare_model_inputs(baseline, candidate)[0]


@pytest.mark.parametrize("allow", [False, True])
def test_cli_requires_separate_explicit_override_for_input_removal(monkeypatch, tmp_path, allow):
    import json
    import sys

    from scripts.validate_registry import main

    baseline = _input_registry()
    candidate = copy.deepcopy(baseline)
    candidate["models"][0]["inputs"].pop(0)
    for name, registry in [("baseline", baseline), ("candidate", candidate)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(registry))
    argv = ["validate_registry.py", str(tmp_path / "candidate.json"), "--baseline",
            str(tmp_path / "baseline.json"), "--min-models", "1", "--allow-large-change"]
    if allow:
        argv.append("--allow-input-removal")
    monkeypatch.setattr(sys, "argv", argv)
    if allow:
        main()
    else:
        with pytest.raises(RegistryValidationError, match="removes existing controls"):
            main()
