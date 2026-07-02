"""Fal Job Inbox: list async fal jobs recorded across ComfyUI sessions."""

from __future__ import annotations

from typing import Any

from .fal_utils import JobStore, logger

_CATEGORY = "FAL/Platform"

_STATUS_CHOICES = ("all", "submitted", "collected")


class FalJobInbox:
    """List async fal jobs from the persistent store; jobs survive restarts."""

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("report", "latest_request_id", "latest_endpoint")
    FUNCTION = "inbox"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Inbox of async fal jobs recorded by Fal Submit. The store is "
        "persistent, so jobs queued in a previous session survive a ComfyUI "
        "restart: submit tonight, restart, then wire latest_request_id and "
        "latest_endpoint into Fal Result by Request ID to collect tomorrow "
        "without re-paying. Never fails the graph."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "status_filter": (
                    list(_STATUS_CHOICES),
                    {
                        "default": "all",
                        "tooltip": (
                            "Which jobs to list: 'submitted' shows jobs still "
                            "waiting to be collected (including ones queued "
                            "before a restart), 'collected' shows finished ones."
                        ),
                    },
                ),
                "limit": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Maximum number of jobs to list, newest first",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        # The job store mutates outside the graph; always re-run.
        return float("nan")

    @staticmethod
    def _filtered_report(store: JobStore, status: str, limit: int) -> str:
        """Plain listing of jobs with one status (report() covers 'all')."""
        entries = store.entries(limit=limit, status=status)
        lines = [f"Fal job inbox ({status}): {len(entries)} shown, newest first"]
        lines.extend(
            f"  {entry.get('endpoint') or '(unknown endpoint)'}  "
            f"req={entry.get('request_id') or '-'}"
            for entry in entries
        )
        return "\n".join(lines)

    def inbox(self, status_filter: str = "all", limit: int = 20) -> tuple[str, str, str]:
        try:
            store = JobStore()
            if status_filter in _STATUS_CHOICES[1:]:
                report = self._filtered_report(store, status_filter, int(limit))
            else:
                report = store.report(limit=int(limit))
            latest = next(iter(store.pending(limit=1)), None)
            latest_request_id = str(latest.get("request_id") or "") if latest else ""
            latest_endpoint = str(latest.get("endpoint") or "") if latest else ""
            return (report, latest_request_id, latest_endpoint)
        except Exception as exc:  # This node must never fail the graph.
            logger.warning("FalJobInbox: could not read job store: %s", exc)
            return ("Fal job inbox: report unavailable", "", "")


NODE_CLASS_MAPPINGS = {
    "FalJobInbox_fal": FalJobInbox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalJobInbox_fal": "Fal Job Inbox (fal)",
}
