"""Deterministic JSON command-line surface for the finite-algebra atlas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.atlas.cli import main
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.serialization import census_spec_canonical_bytes
from anyalgebra.census.spec import CensusSpecCore


def _spec() -> CensusSpec:
    core = CensusSpecCore.create(carrier_size=2, arity=2, corpus_name="cli-order-two")
    return CensusSpec.create(
        carrier_size=2,
        arity=2,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100,
            max_orbits=100,
            max_work_units=1_000,
            max_memory_bytes=1_000_000,
        ),
    )


def _spec_path(tmp_path: Path) -> Path:
    path = tmp_path / "spec.json"
    path.write_bytes(census_spec_canonical_bytes(_spec()))
    return path


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    parsed = json.loads(captured.out)
    assert type(parsed) is dict
    return cast(dict[str, object], parsed)


def test_validate_build_verify_and_query_commands_emit_canonical_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _spec_path(tmp_path)
    assert main(["validate-spec", str(spec)]) == 0
    validated = _output(capsys)
    assert validated == {
        "arity": 2,
        "candidateCount": 16,
        "carrierSize": 2,
        "corpusName": "cli-order-two",
        "specHash": str(_spec().semantic_hash),
    }

    atlas = tmp_path / "atlas"
    assert main(["build", "--spec", str(spec), "--output", str(atlas)]) == 0
    built = _output(capsys)
    assert built["status"] == "complete"
    assert built["corpusCount"] == 1
    assert built["representativeCount"] == 10

    assert main(["verify", "--atlas", str(atlas)]) == 0
    verified = _output(capsys)
    assert verified["atlasHash"] == built["atlasHash"]
    assert verified["verifiedObjectCount"] == built["objectCount"]

    assert (
        main(
            [
                "query",
                "--atlas",
                str(atlas),
                "--carrier-size",
                "2",
                "--arity",
                "2",
                "--law",
                "associativity",
                "--invariant",
                "center_size=2",
            ]
        )
        == 0
    )
    queried = _output(capsys)
    assert queried["examinedRepresentativeCount"] == 10
    matches = queried["matches"]
    match_count = queried["matchCount"]
    assert type(matches) is list
    assert type(match_count) is int
    typed_matches = cast(list[dict[str, object]], matches)
    assert match_count == len(typed_matches)
    assert match_count > 0
    assert [item["canonicalId"] for item in typed_matches] == sorted(
        cast(str, item["canonicalId"]) for item in typed_matches
    )


def test_build_requires_output_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _spec_path(tmp_path)
    assert main(["build", "--spec", str(spec)]) == 2
    missing = capsys.readouterr()
    assert missing.out == ""
    assert json.loads(missing.err) == {"error": "invalid_arguments"}

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "user.txt"
    marker.write_text("preserve", encoding="utf-8")
    assert main(["build", "--spec", str(spec), "--output", str(existing)]) == 2
    failed = capsys.readouterr()
    assert failed.out == ""
    assert json.loads(failed.err) == {"error": "build_rejected"}
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert str(existing) not in failed.err


def test_malformed_spec_and_query_values_have_sanitized_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "private-name.json"
    malformed.write_text('{"secret":"do-not-echo"}', encoding="utf-8")
    assert main(["validate-spec", str(malformed)]) == 2
    failed = capsys.readouterr()
    assert json.loads(failed.err) == {"error": "spec_rejected"}
    assert "secret" not in failed.err
    assert "private-name" not in failed.err

    spec = _spec_path(tmp_path)
    atlas = tmp_path / "atlas"
    assert main(["build", "--spec", str(spec), "--output", str(atlas)]) == 0
    capsys.readouterr()
    assert main(["query", "--atlas", str(atlas), "--invariant", "bad"]) == 2
    query_error = capsys.readouterr()
    assert json.loads(query_error.err) == {"error": "invalid_query"}


def test_pyproject_registers_the_exact_cli_entrypoint() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'anyalgebra-atlas = "anyalgebra.atlas.cli:main"' in pyproject.read_text(
        encoding="utf-8"
    )
