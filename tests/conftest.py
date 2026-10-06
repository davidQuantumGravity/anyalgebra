"""Shared test locations and immutable fixture namespace declarations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
FIXTURE_ROOT: Final = REPOSITORY_ROOT / "tests" / "fixtures"
FAILURE_ARTIFACT_ROOT: Final = REPOSITORY_ROOT / "tests" / "artifacts"
LEGACY_TEST_ROOT: Final = REPOSITORY_ROOT / "tests" / "legacy"
INTERNAL_RECORD_ROOT: Final = REPOSITORY_ROOT / ".agents" / "project-process"
FIXTURE_NAMESPACES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "canonical": "Hand-checkable exact fixtures and their source anchors.",
        "published": "Pinned transcriptions of externally published sources.",
        "legacy": "Pinned legacy captures used only as non-oracular evidence.",
        "generated": "Deterministically generated fixtures with recorded inputs.",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep every test under ``tests/legacy`` in the legacy evidence lane.

    File placement is the durable boundary.  Requiring each legacy test
    function to repeat the marker allowed source-pinned receipt tests to leak
    into the neutral CI selection.

    Tests marked ``internal_records`` audit the maintainers' process records.
    Those records are not part of the public repository, so the tests skip
    visibly when the directory is absent instead of failing.
    """
    skip_internal = (
        None
        if INTERNAL_RECORD_ROOT.is_dir()
        else pytest.mark.skip(
            reason="internal process records are not distributed with this checkout"
        )
    )
    for item in items:
        if Path(str(item.path)).resolve().is_relative_to(LEGACY_TEST_ROOT):
            item.add_marker(pytest.mark.legacy)
        if skip_internal is not None and "internal_records" in item.keywords:
            item.add_marker(skip_internal)


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    """Return the read-only root reserved for versioned fixture inputs."""
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def fixture_namespaces() -> Mapping[str, Path]:
    """Return the declared fixture namespace paths without creating them."""
    return MappingProxyType({name: FIXTURE_ROOT / name for name in FIXTURE_NAMESPACES})


@pytest.fixture(scope="session")
def failure_artifact_root() -> Path:
    """Return the retained location for failure-only diagnostic artifacts."""
    return FAILURE_ARTIFACT_ROOT
