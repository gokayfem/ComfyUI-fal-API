"""Shared fixtures: load the pack exactly like ComfyUI does (hyphenated dir)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "ComfyUI_fal_API"

# keep the persistent result cache out of the user's real cache dir during tests
os.environ.setdefault(
    "COMFYUI_FAL_API_CACHE_DB",
    str(Path(tempfile.mkdtemp(prefix="fal-api-test-cache-")) / "cache.db"),
)


def _load_package():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(
        PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pack():
    """The fully loaded node pack (static + dynamic mappings)."""
    return _load_package()


def _submodule(name: str):
    _load_package()
    return importlib.import_module(f"{PKG}.{name}")


@pytest.fixture(scope="session")
def schema_to_inputs():
    return _submodule("nodes.dynamic.schema_to_inputs")


@pytest.fixture(scope="session")
def arguments_mod():
    return _submodule("nodes.dynamic.arguments")


@pytest.fixture(scope="session")
def outputs_mod():
    return _submodule("nodes.dynamic.outputs")


@pytest.fixture(scope="session")
def factory_mod():
    return _submodule("nodes.dynamic.factory")


@pytest.fixture(scope="session")
def errors_mod():
    return _submodule("nodes.utils.errors")
