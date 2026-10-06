"""Executable contracts for the bounded v0.1 reference-atlas example."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from anyalgebra.census.reference import REFERENCE_BATCH_LIMIT, enumerate_reference
from anyalgebra.census.serialization import (
    census_spec_canonical_bytes,
    census_spec_from_canonical_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "finite_algebra_atlas.py"
FIXTURE = ROOT / "tests" / "fixtures" / "v01_atlas_spec.json"


def _example() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE))


def test_reference_spec_is_the_exact_canonical_fixture() -> None:
    namespace = _example()
    spec = namespace["reference_spec"]()
    fixture = FIXTURE.read_bytes()
    canonical_fixture = fixture.removesuffix(b"\n")

    assert fixture.endswith(b"\n")
    assert canonical_fixture == census_spec_canonical_bytes(spec)
    assert (
        census_spec_canonical_bytes(census_spec_from_canonical_bytes(canonical_fixture))
        == canonical_fixture
    )
    assert str(spec.semantic_hash) == (
        "sha256:361253fe9e43ad6aac9dadcc3aa8d0fc97572a4d8f58939aa282134d1a596dea"
    )


def test_selected_bound_has_an_independent_count_and_fits_declared_work() -> None:
    spec = _example()["reference_spec"]()
    enumeration = enumerate_reference(spec)

    assert spec.core.input_tuple_count == 3**2 == 9
    assert spec.core.candidate_count == 3**9 == 19_683
    assert 3**9 <= REFERENCE_BATCH_LIMIT < 4**16
    assert enumeration.workspace_bytes_per_candidate == 272
    assert 19_683 * 272 == 5_353_776 <= spec.bounds.max_memory_bytes
    assert enumeration.complete


def test_example_builds_and_reports_the_frozen_reference_atlas(tmp_path: Path) -> None:
    report = cast(
        dict[str, Any],
        _example()["build_reference_atlas"](tmp_path / "reference-atlas"),
    )

    assert report == {
        "analysis": {
            "associativeOrbitCount": 12,
            "commutativeOrbitCount": 129,
            "flexibleOrbitCount": 129,
            "idempotentOrbitCount": 7,
            "uniqueIdentityOrbitCount": 15,
            "uniqueZeroOrbitCount": 15,
        },
        "corpus": {
            "acceptedLabeledCount": 729,
            "commutativeDegreesOfFreedom": 6,
            "examinedCandidateCount": 19_683,
            "independentAcceptedCount": "3^6=729",
            "rawTableCount": "3^9=19683",
            "rejectedLabeledCount": 18_954,
            "status": "complete",
        },
        "evidence": {
            "atlasSemanticHash": (
                "sha256:245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843"
            ),
            "objectCount": 862,
            "specSemanticHash": (
                "sha256:361253fe9e43ad6aac9dadcc3aa8d0fc97572a4d8f58939aa282134d1a596dea"
            ),
            "verifiedObjectCount": 862,
        },
        "orbits": {
            "acceptedMemberCertificateCount": 729,
            "equivalence": "full carrier relabeling",
            "permutationCount": 6,
            "representativeCount": 129,
        },
        "scope": (
            "complete only for binary commutative operation tables on the labeled "
            "carrier {0,1,2}, modulo full carrier relabeling; no broader algebra "
            "classification or physics claim"
        ),
    }


def test_example_is_deterministic_and_command_requires_a_fresh_output(
    tmp_path: Path,
) -> None:
    namespace = _example()
    first = namespace["build_reference_atlas"](tmp_path / "atlas-a")
    second = namespace["build_reference_atlas"](tmp_path / "atlas-b")
    assert first == second

    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output", str(tmp_path / "atlas-cli")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        completed.stdout
        == json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n"
    )

    missing = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 2
    assert "--output" in missing.stderr


def test_example_refuses_to_overwrite_its_published_atlas(tmp_path: Path) -> None:
    build = _example()["build_reference_atlas"]
    target = tmp_path / "atlas"
    build(target)

    with pytest.raises(ValueError, match="must not already exist"):
        build(target)
