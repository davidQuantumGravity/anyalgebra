"""Safe loading, reference verification, and deterministic atlas queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anyalgebra.atlas.build import build_finite_algebra_atlas
from anyalgebra.atlas.query import (
    AtlasQueryError,
    load_finite_algebra_atlas,
    query_atlas,
)
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpecCore


def _spec() -> CensusSpec:
    core = CensusSpecCore.create(carrier_size=2, arity=2, corpus_name="query-order-two")
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


def _built(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "atlas"
    build_finite_algebra_atlas((_spec(),), output_directory=target)
    return target


def test_loader_verifies_all_links_and_queries_size_arity_and_identifier(
    tmp_path: Path,
) -> None:
    loaded = load_finite_algebra_atlas(_built(tmp_path))
    all_matches = query_atlas(loaded, carrier_size=2, arity=2)

    assert loaded.atlas.header.status == "complete"
    assert loaded.verified_object_count == len(loaded.atlas.header.evidence_ids)
    assert all_matches.examined_representative_count == 10
    assert len(all_matches.matches) == 10
    assert tuple(str(item.canonical_id) for item in all_matches.matches) == tuple(
        sorted(str(item.canonical_id) for item in all_matches.matches)
    )

    selected = all_matches.matches[4]
    singleton = query_atlas(loaded, canonical_id=selected.canonical_id)
    assert singleton.matches == (selected,)
    assert query_atlas(loaded, carrier_size=3).matches == ()


def test_law_and_allow_listed_invariant_queries_are_deterministic(
    tmp_path: Path,
) -> None:
    loaded = load_finite_algebra_atlas(_built(tmp_path))
    associative = query_atlas(loaded, laws=("associativity",))
    unique_zero = query_atlas(loaded, invariant_fields=(("zero_status", "unique"),))
    center_two = query_atlas(loaded, invariant_fields=(("center_size", 2),))

    assert associative.matches
    assert unique_zero.matches
    assert center_two.matches
    assert associative == query_atlas(loaded, laws=("associativity",))
    assert all(
        match.law_statuses["associativity"] == "proved_on_complete_grid"
        for match in associative.matches
    )
    assert all(
        match.invariants["zero_status"] == "unique" for match in unique_zero.matches
    )
    assert all(match.invariants["center_size"] == 2 for match in center_two.matches)

    with pytest.raises(AtlasQueryError, match="unknown law"):
        query_atlas(loaded, laws=("fabricated",))
    with pytest.raises(AtlasQueryError, match="unknown invariant"):
        query_atlas(loaded, invariant_fields=(("derivation_dimension", 14),))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record.__setitem__("schemaVersion", 2),
            "unknown atlas version",
        ),
        (lambda record: record.__setitem__("extra", True), "atlas record shape"),
        (
            lambda record: record["header"].__setitem__("representativeCount", 99),
            "atlas content",
        ),
    ],
)
def test_corrupt_extra_and_unknown_atlas_records_fail_closed(
    tmp_path: Path, mutation: object, message: str
) -> None:
    root = _built(tmp_path)
    path = root / "atlas.json"
    record = json.loads(path.read_bytes())
    assert callable(mutation)
    mutation(record)
    path.write_text(
        json.dumps(record, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(AtlasQueryError, match=message):
        load_finite_algebra_atlas(root)


def test_missing_corrupt_and_extra_object_files_fail_closed(tmp_path: Path) -> None:
    root = _built(tmp_path)
    object_paths = sorted((root / "objects").rglob("*.json"))
    missing = object_paths[0]
    missing.unlink()
    with pytest.raises(AtlasQueryError, match="object file set"):
        load_finite_algebra_atlas(root)

    root = _built(tmp_path / "corrupt")
    object_path = sorted((root / "objects").rglob("*.json"))[0]
    object_path.write_bytes(b"{}")
    with pytest.raises(AtlasQueryError, match="object content"):
        load_finite_algebra_atlas(root)

    root = _built(tmp_path / "extra")
    (root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AtlasQueryError, match="object file set"):
        load_finite_algebra_atlas(root)


def test_noncanonical_hostile_and_path_boundaries_fail_closed(tmp_path: Path) -> None:
    root = _built(tmp_path)
    atlas_path = root / "atlas.json"
    record = json.loads(atlas_path.read_bytes())
    atlas_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    with pytest.raises(AtlasQueryError, match="canonical JSON"):
        load_finite_algebra_atlas(root)

    oversized = tmp_path / "oversized"
    oversized.mkdir()
    (oversized / "atlas.json").write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(AtlasQueryError, match="file size"):
        load_finite_algebra_atlas(oversized)
    with pytest.raises(AtlasQueryError, match="must be absolute"):
        load_finite_algebra_atlas(Path("relative"))
