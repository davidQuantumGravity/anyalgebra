"""Deterministic, atomic finite-algebra atlas building."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anyalgebra.atlas.build import AtlasBuildError, build_finite_algebra_atlas
from anyalgebra.atlas.models import atlas_canonical_bytes
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpecCore


def _spec(*, max_candidates: int = 100) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        corpus_name=f"order-two-{max_candidates}",
    )
    return CensusSpec.create(
        carrier_size=2,
        arity=2,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=100,
            max_work_units=1_000,
            max_memory_bytes=1_000_000,
        ),
    )


def _tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )


def test_complete_build_composes_census_analysis_certificates_and_receipt(
    tmp_path: Path,
) -> None:
    target = tmp_path / "atlas"
    result = build_finite_algebra_atlas((_spec(),), output_directory=target)
    record = json.loads(result.atlas_path.read_bytes())

    assert result.atlas.header.status == "complete"
    assert result.atlas.header.corpus_count == 1
    assert result.atlas.header.representative_count == 10
    assert result.atlas.corpora[0].accepted_labeled_count == 16
    assert result.atlas.corpora[0].rejected_labeled_count == 0
    assert result.atlas_path == target / "atlas.json"
    assert result.atlas_path.read_bytes() == atlas_canonical_bytes(result.atlas)
    assert result.object_count == len(result.atlas.header.evidence_ids)
    assert record["header"]["status"] == "complete"
    assert all(
        (target / "objects" / item.algorithm / f"{item.digest}.json").is_file()
        for item in result.atlas.header.evidence_ids
    )


def test_two_builds_have_identical_relative_files_and_bytes(tmp_path: Path) -> None:
    first = build_finite_algebra_atlas((_spec(),), output_directory=tmp_path / "first")
    second = build_finite_algebra_atlas(
        (_spec(),), output_directory=tmp_path / "second"
    )

    assert first.atlas == second.atlas
    assert _tree(first.output_directory) == _tree(second.output_directory)


def test_interrupted_enumeration_is_written_only_as_incomplete(tmp_path: Path) -> None:
    result = build_finite_algebra_atlas(
        (_spec(max_candidates=3),), output_directory=tmp_path / "partial"
    )
    corpus = result.atlas.corpora[0]

    assert result.atlas.header.status == "incomplete"
    assert corpus.complete is False
    assert corpus.examined_candidate_count == 3
    assert corpus.accepted_labeled_count == 3
    assert corpus.rejected_labeled_count == 0
    assert corpus.classification_id is None
    assert corpus.representatives == ()


def test_build_is_atomic_when_analysis_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyalgebra.atlas.build as builder

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("simulated analysis failure")

    monkeypatch.setattr(builder, "assemble_finite_carrier_analysis", fail)
    target = tmp_path / "never-published"
    with pytest.raises(AtlasBuildError, match="build failed before publication"):
        build_finite_algebra_atlas((_spec(),), output_directory=target)
    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_output_boundary_and_input_type_fail_before_writes(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(AtlasBuildError, match="must not already exist"):
        build_finite_algebra_atlas((_spec(),), output_directory=existing)
    with pytest.raises(AtlasBuildError, match="must be absolute"):
        build_finite_algebra_atlas((_spec(),), output_directory=Path("relative"))
    with pytest.raises(AtlasBuildError, match="must contain exact CensusSpec"):
        build_finite_algebra_atlas((object(),), output_directory=tmp_path / "bad")
    with pytest.raises(AtlasBuildError, match="must not be empty"):
        build_finite_algebra_atlas((), output_directory=tmp_path / "empty")
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "empty").exists()
