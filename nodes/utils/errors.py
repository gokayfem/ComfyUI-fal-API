"""Error types and helpers for normalizing fal.ai API failures."""

from __future__ import annotations

from typing import Any, NoReturn


class FalApiError(Exception):
    """Raised when a fal.ai API call (or related processing) fails."""

    def __init__(
        self,
        model_name: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.message = message
        self.status_code = status_code
        formatted = f"[{model_name}] {message}"
        if status_code is not None:
            formatted = f"{formatted} (HTTP {status_code})"
        super().__init__(formatted)


def _flatten_validation_detail(detail: list[Any]) -> str:
    """Flatten a FastAPI validation-error list into a readable string."""
    parts: list[str] = []
    for item in detail:
        if isinstance(item, dict):
            loc = ".".join(str(part) for part in (item.get("loc") or []))
            msg = str(item.get("msg", item))
            parts.append(f"{loc}: {msg}" if loc else msg)
        else:
            parts.append(str(item))
    return "; ".join(parts)


def _detail_to_message(detail: Any) -> str:
    """Convert a response 'detail' payload into a message string."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return _flatten_validation_detail(detail)
    return str(detail)


def _message_from_response(response: Any) -> str | None:
    """Extract a human-readable message from an httpx-like response."""
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict) and "detail" in payload:
        return _detail_to_message(payload["detail"])
    if payload is not None:
        return str(payload)

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def extract_error_message(exc: BaseException) -> tuple[str, int | None]:
    """Extract a readable message and HTTP status code from an exception.

    Duck-types fal_client.FalClientHTTPError (``.status_code`` plus an
    httpx ``.response``) so this works without importing fal_client.
    """
    raw_status = getattr(exc, "status_code", None)
    status_code = raw_status if isinstance(raw_status, int) else None

    response = getattr(exc, "response", None)
    if response is not None:
        message = _message_from_response(response)
        if message:
            return message, status_code

    return str(exc) or exc.__class__.__name__, status_code


def raise_fal_error(model_name: str, exc: Exception) -> NoReturn:
    """Normalize any exception into a FalApiError and raise it."""
    if isinstance(exc, FalApiError):
        raise exc
    message, status_code = extract_error_message(exc)
    raise FalApiError(model_name, message, status_code) from exc
