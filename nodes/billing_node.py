"""Billing nodes: account balance reporting with spend-guard visibility."""

from __future__ import annotations

from typing import Any

from .fal_utils import BillingUtils, SpendGuard, logger

_CATEGORY = "FAL/Platform"

_UNAVAILABLE_HINT = (
    "Hint: the balance API needs an API key with billing access "
    "(Authorization: Key ...); check your key at https://fal.ai/dashboard/keys."
)


def _guard_line(settings: dict[str, float]) -> str:
    """One-line summary of the active spend-guard configuration."""
    budget = settings.get("session_budget_usd") or 0.0
    floor = settings.get("min_balance_usd") or 0.0
    if budget <= 0 and floor <= 0:
        return (
            "Spend guard: disabled (set [spend_guard] session_budget_usd / "
            "min_balance_usd in config.ini to enable)"
        )
    parts = []
    if budget > 0:
        parts.append(f"session budget ${budget:.2f}")
    if floor > 0:
        parts.append(f"min balance ${floor:.2f}")
    return f"Spend guard: {', '.join(parts)}"


def _build_report(balance: float | None, settings: dict[str, float]) -> str:
    """Human-readable balance report. Never raises."""
    if balance is not None:
        lines = [f"fal account balance: ${balance:,.2f}"]
    else:
        lines = ["fal account balance: unavailable", _UNAVAILABLE_HINT]
    return "\n".join([*lines, _guard_line(settings)])


class FalBalance:
    """Report the fal.ai account credit balance and active spend-guard limits."""

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "balance_usd")
    FUNCTION = "check"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Fetch your fal.ai account credit balance and show the active "
        "spend-guard settings. Never fails: balance_usd is -1.0 when the "
        "balance API is unavailable."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "force_refresh": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Bypass the 60s balance cache and query fal again",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        # The balance changes outside the graph; always re-run.
        return float("nan")

    def check(self, force_refresh: bool = False) -> tuple[str, float]:
        try:
            balance = BillingUtils.get_balance(force=bool(force_refresh))
            settings = SpendGuard.settings()
            report = _build_report(balance, settings)
        except Exception as err:  # This node must never fail the graph.
            logger.warning("FalBalance: could not build balance report: %s", err)
            balance = None
            report = f"fal account balance: unavailable ({err})\n{_UNAVAILABLE_HINT}"
        return (report, float(balance) if balance is not None else -1.0)


NODE_CLASS_MAPPINGS = {
    "FalBalance_fal": FalBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalBalance_fal": "Fal Account Balance (fal)",
}
