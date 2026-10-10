"""Observable packaging-contract tests for the neutral package skeleton."""

from __future__ import annotations

import tomllib
from pathlib import Path

import anyalgebra
from anyalgebra import _version


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_uses_one_development_version_source() -> None:
    """Distribution and imported versions come from ``_version.py`` only."""
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        metadata = tomllib.load(file)

    assert metadata["build-system"] == {
        "build-backend": "hatchling.build",
        "requires": ["hatchling>=1.27,<2"],
    }
    assert metadata["project"]["name"] == "anyalgebra"
    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert metadata["project"]["dependencies"] == []
    assert metadata["project"]["license"] == "AGPL-3.0-only"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert metadata["tool"]["hatch"]["version"] == {
        "path": "src/anyalgebra/_version.py"
    }
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"] == {
        "packages": ["src/anyalgebra"]
    }
    assert anyalgebra.__version__ == _version.__version__ == "0.1.9"


def test_neutral_package_exposes_only_version_and_typed_marker() -> None:
    """The skeleton imports without optional mathematics systems or public APIs."""
    assert anyalgebra.__all__ == ["__version__"]
    assert (REPOSITORY_ROOT / "src" / "anyalgebra" / "py.typed").is_file()


def test_repository_contains_the_declared_agpl_license() -> None:
    """The declared SPDX expression is backed by the canonical license text."""
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert license_text.lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE")
    assert "Version 3, 19 November 2007" in license_text
