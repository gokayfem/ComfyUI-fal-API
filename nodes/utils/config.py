"""fal.ai API key/config resolution for the ComfyUI-fal-API node pack."""

from __future__ import annotations

import configparser
import os
import threading
from typing import Any

from .errors import FalApiError
from .logger import logger

_PLACEHOLDER_KEY = "<your_fal_api_key_here>"
_MISSING_KEY_MESSAGE = (
    "FAL_KEY is not configured. Set the FAL_KEY environment variable or add it "
    "to config.ini under the [API] section. Get your API key from "
    "https://fal.ai/dashboard/keys"
)


def _config_path() -> str:
    """Return the path to config.ini at the repo root (one dir above nodes/)."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.dirname(utils_dir)
    repo_root = os.path.dirname(nodes_dir)
    return os.path.join(repo_root, "config.ini")


def _read_config() -> configparser.ConfigParser:
    """Read config.ini; returns an empty parser if the file is absent."""
    parser = configparser.ConfigParser()
    try:
        parser.read(_config_path())
    except configparser.Error as exc:
        logger.warning("Failed to parse config.ini: %s", exc)
    return parser


def _resolve_key(parser: configparser.ConfigParser) -> str | None:
    """Resolve the FAL key: environment first, then config.ini [API] FAL_KEY."""
    env_key = os.environ.get("FAL_KEY")
    if env_key:
        logger.info("Using FAL_KEY from environment")
        return env_key

    config_key = parser.get("API", "FAL_KEY", fallback=None)
    if config_key:
        logger.info("Using FAL_KEY from config.ini")
        return config_key
    return None


def _is_valid_key(key: str | None) -> bool:
    return bool(key) and key != _PLACEHOLDER_KEY


class FalConfig:
    """Singleton holding fal.ai configuration and a cached client."""

    _instance: FalConfig | None = None
    _lock = threading.Lock()

    def __new__(cls) -> FalConfig:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        """Resolve the API key once; never raises at import/construction time."""
        self._parser = _read_config()
        self._key: str | None = _resolve_key(self._parser)
        self._client: Any | None = None

        if not _is_valid_key(self._key):
            logger.warning(_MISSING_KEY_MESSAGE)

    def get_client(self) -> Any:
        """Get or create the cached fal_client SyncClient.

        Raises FalApiError if no valid key is configured.
        """
        if self._client is None:
            if not _is_valid_key(self._key):
                raise FalApiError("config", _MISSING_KEY_MESSAGE)
            from fal_client.client import SyncClient

            self._client = SyncClient(key=self._key)
        return self._client

    def get_key(self) -> str | None:
        """Return the resolved FAL API key (may be None or a placeholder)."""
        return self._key

    def get_setting(self, section: str, name: str, default: Any = None) -> Any:
        """Read an arbitrary config.ini setting, with bool parsing.

        Returns ``default`` if the section or option is absent. Values equal to
        "true"/"false" (case-insensitive) are returned as booleans.
        """
        try:
            value = self._parser.get(section, name)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value
