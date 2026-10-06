"""Integration coverage for coexisting module parents and explicit transport."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from anyalgebra.core.domains import DomainElement, ZZ
from anyalgebra.core.elements import (
    ModuleParentMismatchError,
    SparseElement,
    SparseElementConstructionError,
    SparseElementMap,
    transport_sparse_element,
)
from anyalgebra.core.modules import Basis, FreeModule


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "coexisting_parents.py"


def _module() -> FreeModule:
    """Build one equal-looking but independently constructed module parent."""
    return FreeModule(
        ZZ(),
        Basis(("e0", "e1"), coefficient_domain=ZZ()),
        name="coexisting",
    )


def _integer_payload(coefficient: DomainElement[object] | None) -> int:
    """Return one canonical ZZ payload, treating missing support as zero."""
    if coefficient is None:
        return 0
    if coefficient.parent is not ZZ() or type(coefficient.value) is not int:
        raise TypeError("expected a canonical ZZ coefficient")
    return coefficient.value


def _transformed_coordinates(
    coordinates: Mapping[int, DomainElement[object]],
) -> dict[int, int]:
    """Return a total, visibly non-coordinate-preserving sparse transform."""
    return {
        0: -_integer_payload(coordinates.get(1)),
        1: 3 * _integer_payload(coordinates.get(0)),
    }


def _transform(source: FreeModule, target: FreeModule) -> SparseElementMap:
    """Return one directional map with a visibly non-coordinate image."""

    def forward(value: SparseElement) -> SparseElement:
        return target.element(_transformed_coordinates(value.coordinates()))

    return SparseElementMap(source, target, forward)


def test_independent_equal_looking_parents_require_explicit_transport() -> None:
    """Coexisting parents do not merge coordinate identity or module arithmetic."""
    source = _module()
    target = _module()
    foreign = _module()
    source_value = source.element({0: 2, 1: -1})
    target_value = target.element({0: 2, 1: -1})
    element_map = _transform(source, target)

    assert source is not target
    assert source == target
    assert source.fingerprint() == target.fingerprint()
    assert source.name == target.name == "coexisting"
    assert tuple(source_value.coordinates().items()) == tuple(
        target_value.coordinates().items()
    )
    assert source_value != target_value

    with pytest.raises(SparseElementConstructionError):
        target.element(source_value)
    with pytest.raises(ModuleParentMismatchError) as arithmetic_error:
        source_value.add(target_value)
    assert arithmetic_error.value.operation == "add"
    assert arithmetic_error.value.left_parent is source
    assert arithmetic_error.value.right_parent is target

    image = transport_sparse_element(source_value, element_map)
    assert image.parent is target
    assert image == target.element({0: 1, 1: 6})
    assert tuple(image.coordinates().items()) != tuple(
        source_value.coordinates().items()
    )
    assert image != source_value
    assert image.add(target.element({1: -6})) == target.element({0: 1})

    zero_image = transport_sparse_element(source.zero(), element_map)
    assert zero_image == target.zero()
    assert zero_image.parent is target
    assert zero_image.coordinates() == {}

    for rejected in (target_value, foreign.element({0: 2, 1: -1})):
        with pytest.raises(ModuleParentMismatchError) as transport_error:
            transport_sparse_element(rejected, element_map)
        assert transport_error.value.operation == "transport"
        assert transport_error.value.left_parent is source
        assert transport_error.value.right_parent is rejected.parent


def _subprocess_environment() -> dict[str, str]:
    """Return an environment without inherited Python-installation leakage."""
    environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one checked subprocess and retain output for assertion diagnostics."""
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        text=True,
    )


def _venv_python(directory: Path) -> Path:
    """Create one clean venv and return its cross-platform interpreter path."""
    venv.EnvBuilder(with_pip=True).create(directory)
    candidates = (directory / "Scripts" / "python.exe", directory / "bin" / "python")
    return next(candidate for candidate in candidates if candidate.is_file())


def _assert_example_payload(payload: object) -> None:
    """Assert the complete address-free semantic example output."""
    assert payload == {
        "arithmetic_rejection": "ModuleParentMismatchError",
        "negative_transport": {
            "foreign_parent": "ModuleParentMismatchError",
            "wrong_direction": "ModuleParentMismatchError",
        },
        "parents": {
            "same_display_name": True,
            "same_fingerprint": True,
            "same_instance": False,
            "structurally_equal": True,
        },
        "reinterpretation_rejection": "SparseElementConstructionError",
        "same_coordinates": {
            "python_equal": False,
            "source": [[0, 2], [1, -1]],
            "target": [[0, 2], [1, -1]],
        },
        "scope": "infrastructure example only; no research claim verified",
        "transport": {
            "coordinate_preserving": False,
            "image": [[0, 1], [1, 6]],
            "image_parent_is_target": True,
            "zero_image": [],
            "zero_parent_is_target": True,
        },
    }


@pytest.mark.slow
def test_example_runs_from_source_and_an_isolated_installed_wheel(
    tmp_path: Path,
) -> None:
    """Execute the example against source and only a clean installed wheel."""
    source_environment = _subprocess_environment()
    source_environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    source_result = _run(
        (sys.executable, EXAMPLE_PATH),
        cwd=tmp_path,
        environment=source_environment,
    )
    assert source_result.returncode == 0, source_result.stderr
    source_payload = json.loads(source_result.stdout)
    _assert_example_payload(source_payload)

    build_python = _venv_python(tmp_path / "build-environment")
    wheel_directory = tmp_path / "wheels"
    wheel_directory.mkdir()
    build_result = _run(
        (
            build_python,
            "-m",
            "pip",
            "--isolated",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            wheel_directory,
            REPOSITORY_ROOT,
        ),
        cwd=tmp_path,
        environment=_subprocess_environment(),
    )
    assert build_result.returncode == 0, build_result.stdout + build_result.stderr
    wheels = tuple(wheel_directory.glob("anyalgebra-*.whl"))
    assert len(wheels) == 1

    install_python = _venv_python(tmp_path / "install-environment")
    install_result = _run(
        (
            install_python,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            wheels[0],
        ),
        cwd=tmp_path,
        environment=_subprocess_environment(),
    )
    assert install_result.returncode == 0, install_result.stdout + install_result.stderr

    external_directory = tmp_path / "external-working-directory"
    external_directory.mkdir()
    copied_example = external_directory / EXAMPLE_PATH.name
    shutil.copy2(EXAMPLE_PATH, copied_example)
    installed_result = _run(
        (install_python, "-I", copied_example),
        cwd=external_directory,
        environment=_subprocess_environment(),
    )
    assert installed_result.returncode == 0, installed_result.stderr
    installed_payload = json.loads(installed_result.stdout)
    _assert_example_payload(installed_payload)
    assert installed_payload == source_payload
