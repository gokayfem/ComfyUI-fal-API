"""Shared model/input fixture builders for the dynamic-node tests."""

from __future__ import annotations


def _model(inputs, **overrides):
    base = {
        "endpoint_id": "fal-ai/test/model",
        "title": "Test Model",
        "category": "text-to-image",
        "lab": "Test Lab",
        "family": "",
        "description": "",
        "pricing": "",
        "published_at": "2026-01-01T00:00:00Z",
        "thumbnail": "",
        "inputs": inputs,
        "output_kind": "images",
        "output_props": ["images"],
    }
    return {**base, **overrides}


def _input(name, type_, **kw):
    base = {
        "name": name,
        "type": type_,
        "required": False,
        "default": None,
        "enum": None,
        "min": None,
        "max": None,
        "description": "",
        "media_kind": None,
        "is_list": False,
        "multiline": False,
    }
    return {**base, **kw}
