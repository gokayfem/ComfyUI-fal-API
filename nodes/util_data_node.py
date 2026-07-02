"""Data utility nodes: JSON path extraction, prompt line cycling, text templating."""

from __future__ import annotations

import json
import re
from typing import Any

from .fal_utils import FalApiError, logger

_CATEGORY = "FAL/Utils/Data"

_MAX_INDEX = 2**31 - 1
_MISSING = object()

# a path segment is an optional key name followed by zero or more [N] indices
_PATH_SEGMENT = re.compile(r"^([^\[\]]*)((?:\[\d+\])*)$")
_BRACKET_INDEX = re.compile(r"\[(\d+)\]")


def _tokenize_path(path: str) -> list[Any]:
    """Split a dot/bracket path into str keys and int indices. No eval."""
    tokens: list[Any] = []
    for part in path.split("."):
        segment = part.strip()
        if not segment:
            continue
        match = _PATH_SEGMENT.match(segment)
        if match is None:
            # contract: anything that can't resolve returns the default,
            # a malformed segment included — it can never match a key anyway
            logger.debug("FalJSONExtract: unparseable path segment %r in %r", segment, path)
            return None
        name, brackets = match.group(1), match.group(2)
        if name:
            tokens.append(name)
        tokens.extend(int(index) for index in _BRACKET_INDEX.findall(brackets))
    return tokens


def _value_to_bool(value: Any) -> bool:
    """Truthiness with JSON-string awareness: "false"/"0"/"no"/"" are False."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "none", "null")
    return bool(value)


def _walk_path(value: Any, tokens: list[Any]) -> Any:
    """Follow tokens through nested dicts/lists; return _MISSING when absent."""
    current = value
    for token in tokens:
        index = token if isinstance(token, int) else None
        if index is None and isinstance(current, list) and str(token).isdigit():
            index = int(token)  # bare integer segment indexing an array
        if index is not None:
            if isinstance(current, list) and 0 <= index < len(current):
                current = current[index]
            else:
                return _MISSING
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            return _MISSING
    return current


def _value_to_text(value: Any) -> str:
    """Strings pass through; everything else is re-serialized as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _value_to_number(value: Any) -> float:
    """Coerce a value to float; anything non-numeric becomes 0.0."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


class FalJSONExtract:
    """Pull a value out of a JSON result by dot/bracket path."""

    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("text", "number", "boolean")
    FUNCTION = "extract"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Extract a value from JSON text by path (e.g. video.url, "
        "images[0].url). Returns it as text, number, and boolean so it can "
        "wire straight into other nodes. Missing paths return the default "
        "instead of failing the graph."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "json_text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": (
                            "Wire result_json from a Fal Any Endpoint / Fal Collect "
                            "node here to pick values out of the raw API result."
                        ),
                    },
                ),
                "path": (
                    "STRING",
                    {
                        "default": "video.url",
                        "tooltip": (
                            "Dot/bracket path into the JSON, e.g. video.url, "
                            "images[0].url, data.items[2].name. Bare integers "
                            "also index arrays (images.0.url)."
                        ),
                    },
                ),
                "default": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Returned as text when the path is missing (not an error)",
                    },
                ),
            },
        }

    def extract(self, json_text: str, path: str = "video.url", default: str = "") -> tuple[str, float, bool]:
        try:
            payload = json.loads(json_text)
        except Exception as exc:
            logger.error("FalJSONExtract: invalid JSON input: %s", exc)
            raise FalApiError("FalJSONExtract", f"Input is not valid JSON: {exc}") from exc

        tokens = _tokenize_path(path or "")
        value = _walk_path(payload, tokens) if tokens is not None else _MISSING
        if value is _MISSING:
            logger.debug("FalJSONExtract: path %r missing, returning default", path)
            value = default
        return (_value_to_text(value), _value_to_number(value), _value_to_bool(value))


class FalPromptLines:
    """Cycle through a multiline prompt list, one line per run."""

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("line", "index", "total")
    FUNCTION = "pick"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Pick one line from a multiline text by index. The index wraps "
        "around (modulo the number of lines), so with control_after_generate "
        "set to increment it cycles through your prompt list forever."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt list, one prompt per line",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MAX_INDEX,
                        "control_after_generate": True,
                        "tooltip": (
                            "Which line to pick (wraps around). Set the control to "
                            "'increment' to iterate through your prompt list run by run."
                        ),
                    },
                ),
                "skip_blank": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Ignore empty/whitespace-only lines"},
                ),
            },
        }

    def pick(self, text: str, index: int = 0, skip_blank: bool = True) -> tuple[str, int, int]:
        lines = (text or "").splitlines()
        if skip_blank:
            lines = [line for line in lines if line.strip()]
        total = len(lines)
        if total == 0:
            return ("", 0, 0)
        effective = int(index) % total
        return (lines[effective], effective, total)


class FalTextTemplate:
    """Fill a text template's {a}..{d} placeholders from string inputs."""

    RETURN_TYPES = ("STRING",)
    FUNCTION = "render"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Substitute {a}, {b}, {c}, {d} placeholders in a template with the "
        "connected string inputs — quick prompt assembly without string "
        "concatenation chains. Missing inputs become empty text."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "template": (
                    "STRING",
                    {
                        "default": "a photo of {a}, {b} style",
                        "multiline": True,
                        "tooltip": "Template text; {a} {b} {c} {d} are replaced with the inputs below",
                    },
                ),
            },
            "optional": {
                "a": ("STRING", {"default": "", "tooltip": "Value for {a}"}),
                "b": ("STRING", {"default": "", "tooltip": "Value for {b}"}),
                "c": ("STRING", {"default": "", "tooltip": "Value for {c}"}),
                "d": ("STRING", {"default": "", "tooltip": "Value for {d}"}),
            },
        }

    def render(self, template: str, a: str = "", b: str = "", c: str = "", d: str = "") -> tuple[str]:
        result = template or ""
        for key, value in (("a", a), ("b", b), ("c", c), ("d", d)):
            result = result.replace("{" + key + "}", value or "")
        return (result,)


NODE_CLASS_MAPPINGS = {
    "FalJSONExtract_fal": FalJSONExtract,
    "FalPromptLines_fal": FalPromptLines,
    "FalTextTemplate_fal": FalTextTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalJSONExtract_fal": "JSON Extract (fal)",
    "FalPromptLines_fal": "Prompt Lines (fal)",
    "FalTextTemplate_fal": "Text Template (fal)",
}
