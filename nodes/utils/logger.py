"""Shared logger for the ComfyUI-fal-API node pack."""

from __future__ import annotations

import logging

_LOGGER_NAME = "ComfyUI-fal-API"
_LOG_FORMAT = "[%(name)s] %(levelname)s: %(message)s"


def _configure_logger() -> logging.Logger:
    """Configure the package logger exactly once."""
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


logger = _configure_logger()
