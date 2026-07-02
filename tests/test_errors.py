"""Unit tests for fal error extraction and FalApiError formatting."""

from __future__ import annotations

import pytest


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeHTTPError(Exception):
    """Duck-typed stand-in for fal_client.FalClientHTTPError."""

    def __init__(self, message, status_code, payload):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(payload)


def test_error_message_includes_model_and_status(errors_mod):
    err = errors_mod.FalApiError("fal-ai/flux/dev", "boom", 422)
    assert "fal-ai/flux/dev" in str(err)
    assert "boom" in str(err)
    assert "422" in str(err)


def test_extract_string_detail(errors_mod):
    exc = _FakeHTTPError("HTTP 403", 403, {"detail": "Content policy violation"})
    message, status = errors_mod.extract_error_message(exc)
    assert message == "Content policy violation"
    assert status == 403


def test_extract_validation_list(errors_mod):
    exc = _FakeHTTPError(
        "HTTP 422", 422,
        {"detail": [
            {"loc": ["body", "prompt"], "msg": "field required"},
            {"loc": ["body", "seed"], "msg": "not an int"},
        ]},
    )
    message, status = errors_mod.extract_error_message(exc)
    assert "prompt: field required" in message
    assert "seed: not an int" in message
    assert status == 422


def test_extract_falls_back_to_str(errors_mod):
    message, status = errors_mod.extract_error_message(RuntimeError("plain failure"))
    assert message == "plain failure"
    assert status is None


def test_extract_survives_bad_response_json(errors_mod):
    exc = _FakeHTTPError("HTTP 500", 500, ValueError("not json"))
    message, status = errors_mod.extract_error_message(exc)
    assert message  # falls back to str(exc)
    assert status == 500


def test_raise_fal_error_chains(errors_mod):
    original = RuntimeError("root cause")
    with pytest.raises(errors_mod.FalApiError) as excinfo:
        errors_mod.raise_fal_error("some-model", original)
    assert excinfo.value.__cause__ is original


def test_raise_fal_error_passthrough(errors_mod):
    already = errors_mod.FalApiError("m", "msg")
    with pytest.raises(errors_mod.FalApiError) as excinfo:
        errors_mod.raise_fal_error("other", already)
    assert excinfo.value is already
