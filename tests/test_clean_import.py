"""Packaging tests for a core installation with no optional backends."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TOOL_PATH = REPOSITORY_ROOT / "tools" / "clean_install_smoke.py"


def _load_smoke_tool() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "clean_install_smoke", SMOKE_TOOL_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_backends_reference_without_optionals_clean_install_imports_wheel() -> None:
    """The installed core imports from its wheel without optional systems."""
    smoke_tool = _load_smoke_tool()

    result = smoke_tool.run_clean_install_smoke(REPOSITORY_ROOT)

    assert result.version == "0.1.9"
    assert result.package_location.endswith("anyalgebra/__init__.py")
    assert result.package_location.startswith(result.purelib_location)
    assert result.optional_modules_attempted == ()
    assert result.optional_modules_imported == ()


def test_find_single_wheel_rejects_missing_wheel(tmp_path: Path) -> None:
    """A missing build artifact has a distinct, deterministic failure."""
    smoke_tool = _load_smoke_tool()

    with pytest.raises(smoke_tool.SmokeFailure, match="build failed: no wheel"):
        smoke_tool.find_single_wheel(tmp_path)


def test_find_single_wheel_rejects_ambiguous_wheel_set(tmp_path: Path) -> None:
    """An ambiguous wheel directory cannot silently choose an artifact."""
    smoke_tool = _load_smoke_tool()
    (tmp_path / "anyalgebra-0.0.2-py3-none-any.whl").touch()
    (tmp_path / "anyalgebra-0.0.1-py3-none-any.whl").touch()

    with pytest.raises(
        smoke_tool.SmokeFailure, match="build failed: expected one wheel"
    ):
        smoke_tool.find_single_wheel(tmp_path)


def test_optional_import_audit_rejects_and_records_attempt(tmp_path: Path) -> None:
    """The import boundary detects a caught-or-uncaught optional import attempt."""
    smoke_tool = _load_smoke_tool()
    denied_root = "anyalgebra_missing_optional_backend"
    audit = smoke_tool.OptionalImportAudit((denied_root,))
    sys.meta_path.insert(0, audit)
    try:
        with pytest.raises(ImportError, match="unexpected optional import"):
            importlib.util.find_spec(denied_root)
    finally:
        sys.meta_path.remove(audit)

    assert audit.attempted_roots == [denied_root]
