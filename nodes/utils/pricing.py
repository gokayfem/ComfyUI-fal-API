"""Pricing parsing and cost estimation from the committed fal model registry."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from .logger import logger

# Units that imply one billable output per run (single-output assumption).
_RUN_UNITS = frozenset({"image", "video", "generation", "request", "run"})

# Tokens that terminate a unit phrase ("$0.015 per megapixel in TURBO mode").
_UNIT_STOPWORDS = frozenset(
    {
        "a",
        "along",
        "an",
        "and",
        "are",
        "at",
        "each",
        "for",
        "if",
        "in",
        "is",
        "on",
        "or",
        "per",
        "plus",
        "rounded",
        "the",
        "to",
        "when",
        "will",
        "with",
        "without",
    }
)

_MONEY = r"([\d,]+(?:\.\d+)?)"

# "For $1.00, you can run this model (with approximately) 12 times"
_RUNS_RATIO_RE = re.compile(
    rf"for\s+\${_MONEY},?\s+you\s+can\s+run\s+this\s+model\s+"
    rf"(?:with\s+)?(?:approximately\s+)?{_MONEY}\s+times",
    re.IGNORECASE,
)

# "$0.05 per second of video", "$0.025 per 1000 characters", "$0.08 per image"
_PER_UNIT_RE = re.compile(
    rf"\$\s*{_MONEY}\s+per\s+(\w+(?:\s+\w+){{0,3}})",
    re.IGNORECASE,
)

_registry_lock = threading.Lock()
_pricing_map: dict[str, str] | None = None


def _registry_path() -> str:
    """Return the path to data/fal_registry.json at the repo root."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.dirname(utils_dir)
    repo_root = os.path.dirname(nodes_dir)
    return os.path.join(repo_root, "data", "fal_registry.json")


def _load_pricing_map() -> dict[str, str]:
    """Lazily load the endpoint_id -> pricing-text map (cached module-wide)."""
    global _pricing_map
    if _pricing_map is not None:
        return _pricing_map
    with _registry_lock:
        if _pricing_map is not None:
            return _pricing_map
        mapping: dict[str, str] = {}
        try:
            with open(_registry_path(), encoding="utf-8") as fh:
                registry = json.load(fh)
            for model in registry.get("models") or []:
                endpoint_id = model.get("endpoint_id")
                if endpoint_id:
                    mapping[endpoint_id] = str(model.get("pricing") or "")
        except Exception as exc:
            logger.warning("Could not load pricing registry: %s", exc)
        _pricing_map = mapping
        return _pricing_map


def _to_float(text: str) -> float:
    """Parse a dollar/count figure, tolerating thousands separators."""
    return float(text.replace(",", ""))


def _clean_unit(phrase: str) -> str | None:
    """Trim a raw unit capture to the meaningful phrase, or None if empty."""
    tokens = phrase.lower().split()
    kept: list[str] = []
    for position, token in enumerate(tokens):
        stripped = token.strip(".,")
        if stripped in _UNIT_STOPWORDS:
            break
        if stripped == "of":
            has_object = (
                bool(kept)
                and position + 1 < len(tokens)
                and tokens[position + 1].strip(".,") not in _UNIT_STOPWORDS
            )
            if not has_object:
                break
        kept.append(stripped)
    cleaned = " ".join(kept)
    return cleaned or None


def _head_noun(unit: str) -> str:
    """Return the singular head noun of a unit phrase.

    "second of video" -> "second"; "generated image" -> "image";
    "image generated" -> "image" (trailing participles are dropped).
    """
    tokens = unit.split(" of ")[0].split()
    while len(tokens) > 1 and tokens[-1].endswith("ed"):
        tokens = tokens[:-1]
    head = tokens[-1]
    if head.endswith("s") and not head.endswith("ss"):
        return head[:-1]
    return head


def _format_amount(value: float) -> str:
    """Format a dollar amount compactly (up to 4 decimals, no trailing zeros)."""
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _empty_parse(raw: str) -> dict[str, Any]:
    return {"per_run": None, "per_unit": None, "unit": None, "raw": raw}


class PricingUtils:
    """Best-effort cost estimation from the registry's human pricing strings."""

    @staticmethod
    def parse(pricing_text: str) -> dict[str, Any]:
        """Parse a human pricing string into structured numbers.

        Precedence for ``per_run``: the explicit "For $A, you can run this
        model N times" ratio wins over a "$X per <unit>" price; the latter
        only sets ``per_run`` for single-output units (image, video,
        generation, request, run). Never raises; unmatched strings return
        all-None fields with ``raw`` preserved.
        """
        raw = pricing_text or ""
        result = _empty_parse(raw)
        try:
            ratio = _RUNS_RATIO_RE.search(raw)
            if ratio:
                runs = _to_float(ratio.group(2))
                if runs > 0:
                    result = {**result, "per_run": _to_float(ratio.group(1)) / runs}

            per_unit = _PER_UNIT_RE.search(raw)
            if per_unit:
                unit = _clean_unit(per_unit.group(2))
                if unit:
                    result = {
                        **result,
                        "per_unit": _to_float(per_unit.group(1)),
                        "unit": unit,
                    }
                    if result["per_run"] is None and _head_noun(unit) in _RUN_UNITS:
                        result = {**result, "per_run": result["per_unit"]}
        except Exception as exc:
            logger.debug("Failed to parse pricing text %r: %s", raw, exc)
        return result

    @staticmethod
    def pricing_for(endpoint_id: str) -> dict[str, Any] | None:
        """Parsed pricing for an endpoint, or None if unknown/unpublished."""
        pricing_text = _load_pricing_map().get(endpoint_id)
        if not pricing_text:
            return None
        return PricingUtils.parse(pricing_text)

    @staticmethod
    def estimate(endpoint_id: str, runs: int = 1) -> dict[str, Any]:
        """Estimate the cost of ``runs`` runs of an endpoint."""
        parsed = PricingUtils.pricing_for(endpoint_id) or _empty_parse("")
        per_run = parsed["per_run"]
        unit_note = ""
        if parsed["per_unit"] is not None and parsed["unit"]:
            unit_note = f"${_format_amount(parsed['per_unit'])} per {parsed['unit']}"
        return {
            "endpoint_id": endpoint_id,
            "runs": runs,
            "per_run": per_run,
            "unit_note": unit_note,
            "total": per_run * runs if per_run is not None else None,
            "raw": parsed["raw"],
        }

    @staticmethod
    def format_report(estimate: dict[str, Any]) -> str:
        """Render an estimate as a compact human string. Never raises."""
        try:
            endpoint_id = estimate.get("endpoint_id") or "unknown"
            runs = estimate.get("runs") or 1
            per_run = estimate.get("per_run")
            total = estimate.get("total")
            unit_note = estimate.get("unit_note") or ""

            if per_run is not None:
                report = f"{endpoint_id}: ~${_format_amount(per_run)}/run"
                if runs != 1 and total is not None:
                    report += f" → ~${_format_amount(total)} for {runs} runs"
                return report
            if unit_note:
                depends_on = (
                    "duration"
                    if re.search(r"\b(second|minute|hour)s?\b", unit_note)
                    else "output size"
                )
                return f"{endpoint_id}: {unit_note} (per-run total depends on {depends_on})"
            return f"{endpoint_id}: pricing not published"
        except Exception as exc:
            logger.debug("format_report failed: %s", exc)
            return "pricing not published"
