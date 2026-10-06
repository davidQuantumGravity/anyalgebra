"""Adversarial boundaries for the bounded v0.1 census and atlas surfaces."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from anyalgebra.atlas.build import AtlasBuildError, build_finite_algebra_atlas
from anyalgebra.atlas.query import (
    AtlasQueryError,
    load_finite_algebra_atlas,
    query_atlas,
)
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.serialization import (
    census_spec_canonical_bytes,
    census_spec_from_canonical_bytes,
    census_spec_from_record,
    census_spec_record,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.persistence.registry import SchemaError


def _spec() -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=1,
        arity=2,
        corpus_name="v01-hostile-boundary",
    )
    return CensusSpec.create(
        carrier_size=1,
        arity=2,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=1,
            max_orbits=1,
            max_work_units=100,
            max_memory_bytes=100_000,
        ),
    )


def _built(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "atlas"
    build_finite_algebra_atlas((_spec(),), output_directory=root)
    return root


class _Bomb:
    def __repr__(self) -> str:
        raise AssertionError("repr payload must not run")

    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("equality payload must not run")

    def __hash__(self) -> int:
        raise AssertionError("hash payload must not run")


class _BoundedProbe:
    """Yield harmless items until one read beyond the public limit is attempted."""

    def __init__(self, *, value: object, allowed_reads: int) -> None:
        self.value = value
        self.allowed_reads = allowed_reads
        self.reads = 0

    def __iter__(self) -> Iterator[object]:
        while True:
            self.reads += 1
            if self.reads > self.allowed_reads:
                raise RuntimeError("private iterator payload")
            yield self.value


def test_build_bounds_hostile_spec_iterable_before_writes(tmp_path: Path) -> None:
    probe = _BoundedProbe(value=_spec(), allowed_reads=257)
    target = tmp_path / "must-not-exist"

    with pytest.raises(AtlasBuildError, match="item limit exceeded") as caught:
        build_finite_algebra_atlas(probe, output_directory=target)

    assert probe.reads == 257
    assert not target.exists()
    assert "private iterator payload" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_query_bounds_hostile_iterable_without_overconsumption(tmp_path: Path) -> None:
    loaded = load_finite_algebra_atlas(_built(tmp_path))
    probe = _BoundedProbe(value="associativity", allowed_reads=257)

    with pytest.raises(AtlasQueryError, match="item limit exceeded") as caught:
        query_atlas(loaded, laws=probe)

    assert probe.reads == 257
    assert "private iterator payload" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_iterator_exception_is_typed_sanitized_and_atomic(tmp_path: Path) -> None:
    def hostile() -> Iterator[CensusSpec]:
        yield _spec()
        raise RuntimeError("caller secret 8f0d18")

    target = tmp_path / "no-partial-atlas"
    with pytest.raises(AtlasBuildError, match="iterable snapshot failed") as caught:
        build_finite_algebra_atlas(hostile(), output_directory=target)

    assert "8f0d18" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not target.exists()


def test_hostile_values_are_rejected_without_repr_equality_or_hash(
    tmp_path: Path,
) -> None:
    target = tmp_path / "invalid-spec"
    with pytest.raises(AtlasBuildError, match="exact CensusSpec"):
        build_finite_algebra_atlas((_Bomb(),), output_directory=target)
    assert not target.exists()

    with pytest.raises(SchemaError, match="invalid_record"):
        census_spec_from_record(_Bomb())


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schemaType":"anyalgebra.census.spec","schemaType":"duplicate"}',
        b'{"schemaType":"anyalgebra.census.spec","schemaVersion":1e999}',
        b'{"schemaType":"anyalgebra.census.spec","schemaVersion":NaN}',
        b'"__import__(\\"os\\").system(\\"never\\")"',
        b"\xff",
        b" " * (1_048_576 + 1),
    ),
    ids=("duplicate-key", "nonexact-number", "nan", "code-shaped", "utf8", "size"),
)
def test_malformed_serialized_specs_fail_closed(payload: bytes) -> None:
    with pytest.raises(SchemaError, match="invalid_census_spec_bytes") as caught:
        census_spec_from_canonical_bytes(payload)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_serialized_code_shaped_text_is_data_and_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    record = census_spec_record(_spec())
    record["corpusName"] = f'__import__("pathlib").Path({str(marker)!r}).touch()'

    with pytest.raises(SchemaError, match="parser_failed"):
        census_spec_from_record(record)

    assert not marker.exists()


def test_unknown_hash_path_components_cannot_escape_object_store(
    tmp_path: Path,
) -> None:
    root = _built(tmp_path)
    atlas_path = root / "atlas.json"
    record = json.loads(atlas_path.read_bytes())
    record["header"]["evidenceIds"][0]["digest"] = "../../outside"
    atlas_path.write_text(
        json.dumps(record, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(AtlasQueryError, match="atlas content drift") as caught:
        load_finite_algebra_atlas(root)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not (tmp_path / "outside").exists()


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable on this platform: {type(error).__name__}")


def test_symlink_root_and_atlas_file_fail_closed(tmp_path: Path) -> None:
    root = _built(tmp_path / "real")
    linked_root = tmp_path / "linked-root"
    _symlink_or_skip(linked_root, root, directory=True)
    with pytest.raises(AtlasQueryError, match="symlink root"):
        load_finite_algebra_atlas(linked_root)

    atlas_path = root / "atlas.json"
    outside = tmp_path / "outside-atlas.json"
    outside.write_bytes(atlas_path.read_bytes())
    atlas_path.unlink()
    _symlink_or_skip(atlas_path, outside)
    with pytest.raises(AtlasQueryError, match="symlink file"):
        load_finite_algebra_atlas(root)


def test_symlink_object_cannot_escape_atlas_root(tmp_path: Path) -> None:
    root = _built(tmp_path / "real")
    object_path = sorted((root / "objects").rglob("*.json"))[0]
    outside = tmp_path / "outside-object.json"
    outside.write_bytes(object_path.read_bytes())
    object_path.unlink()
    _symlink_or_skip(object_path, outside)

    with pytest.raises(AtlasQueryError, match="path escapes atlas root"):
        load_finite_algebra_atlas(root)


def test_environment_cannot_turn_serialized_text_into_an_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "environment-executed"
    monkeypatch.setenv(
        "ANYALGEBRA_IMPORT",
        f'__import__("pathlib").Path({str(marker)!r}).touch()',
    )

    assert (
        census_spec_from_canonical_bytes(census_spec_canonical_bytes(_spec()))
        == _spec()
    )
    assert not marker.exists()
    assert os.environ["ANYALGEBRA_IMPORT"].startswith("__import__")
