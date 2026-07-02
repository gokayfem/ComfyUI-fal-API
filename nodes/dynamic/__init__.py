"""Dynamic fal.ai node package: auto-generated nodes from the model registry."""

from __future__ import annotations


def get_dynamic_mappings() -> tuple[dict[str, type], dict[str, str]]:
    """Return (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS) for dynamic nodes.

    Never raises: any failure (missing registry, missing utils facade, bad
    schema) yields empty mappings so static node loading is never affected.
    """
    try:
        from .registry_loader import load_dynamic_mappings

        return load_dynamic_mappings()
    except Exception as err:
        import logging

        logging.getLogger(__name__).error(
            "Failed to load dynamic fal nodes: %s", err
        )
        return {}, {}


__all__ = ["get_dynamic_mappings"]
