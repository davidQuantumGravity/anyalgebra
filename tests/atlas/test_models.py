"""Version-one finite-algebra atlas record contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.atlas.models import (
    AtlasError,
    AtlasRepresentative,
    FiniteAlgebraAtlas,
    atlas_canonical_bytes,
    atlas_record,
    create_atlas_corpus,
)
from anyalgebra.core.parents import SemanticHash


def _hash(digit: str) -> SemanticHash:
    return SemanticHash("sha256", digit * 64)


def _representative(digit: str, outputs: tuple[int, ...]) -> AtlasRepresentative:
    return AtlasRepresentative.create(
        canonical_id=_hash(digit),
        representative_outputs=outputs,
        full_orbit_size=2,
        accepted_member_count=2,
        certificate_ids=(_hash("a"), _hash("b")),
        analysis_ids=(_hash("c"),),
    )


def test_complete_atlas_is_canonical_cross_linked_and_content_addressed() -> None:
    second = _representative("2", (1, 0, 0, 1))
    first = _representative("1", (0, 0, 1, 1))
    corpus = create_atlas_corpus(
        corpus_id="order-two-binary",
        spec_id=_hash("3"),
        enumeration_id=_hash("4"),
        classification_id=_hash("5"),
        receipt_id=_hash("6"),
        total_candidate_count=4,
        examined_candidate_count=4,
        accepted_labeled_count=4,
        rejected_labeled_count=0,
        representatives=(second, first),
        complete=True,
    )
    atlas = FiniteAlgebraAtlas.create((corpus,))
    replay = FiniteAlgebraAtlas.create((corpus,))
    record = atlas_record(atlas)

    assert atlas == replay
    assert atlas.header.schema_version == 1
    assert atlas.header.status == "complete"
    assert atlas.header.corpus_count == 1
    assert atlas.header.representative_count == 2
    assert tuple(item.canonical_id for item in corpus.representatives) == (
        _hash("1"),
        _hash("2"),
    )
    content_hash = record["contentHash"]
    assert type(content_hash) is dict
    assert content_hash["digest"] == atlas.semantic_hash.digest
    assert json.loads(atlas_canonical_bytes(atlas))["schemaVersion"] == 1
    assert atlas_canonical_bytes(atlas) == atlas_canonical_bytes(replay)


def test_incomplete_corpus_cannot_promote_or_carry_classified_orbits() -> None:
    corpus = create_atlas_corpus(
        corpus_id="bounded-prefix",
        spec_id=_hash("1"),
        enumeration_id=_hash("2"),
        classification_id=None,
        receipt_id=_hash("3"),
        total_candidate_count=16,
        examined_candidate_count=3,
        accepted_labeled_count=2,
        rejected_labeled_count=1,
        representatives=(),
        complete=False,
    )
    atlas = FiniteAlgebraAtlas.create((corpus,))

    assert atlas.header.status == "incomplete"
    assert atlas.header.representative_count == 0
    assert corpus.accepted_labeled_count == 2
    with pytest.raises(AtlasError, match="complete corpus must examine"):
        create_atlas_corpus(
            corpus_id="false-promotion",
            spec_id=_hash("1"),
            enumeration_id=_hash("2"),
            classification_id=_hash("4"),
            receipt_id=_hash("3"),
            total_candidate_count=16,
            examined_candidate_count=3,
            accepted_labeled_count=0,
            rejected_labeled_count=3,
            representatives=(),
            complete=True,
        )
    with pytest.raises(AtlasError, match="incomplete corpus cannot carry"):
        create_atlas_corpus(
            corpus_id="partial-orbits",
            spec_id=_hash("1"),
            enumeration_id=_hash("2"),
            classification_id=_hash("4"),
            receipt_id=_hash("3"),
            total_candidate_count=16,
            examined_candidate_count=3,
            accepted_labeled_count=2,
            rejected_labeled_count=1,
            representatives=(_representative("5", (0, 0, 0, 0)),),
            complete=False,
        )


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("accepted_labeled_count", (3, 0), "accepted plus rejected"),
        ("accepted_labeled_count", (4, 0), "representatives do not account"),
    ],
)
def test_count_equations_fail_before_construction(
    field: str, values: tuple[int, int], message: str
) -> None:
    del field
    accepted, rejected = values
    with pytest.raises(AtlasError, match=message):
        create_atlas_corpus(
            corpus_id="bad-accounting",
            spec_id=_hash("1"),
            enumeration_id=_hash("2"),
            classification_id=_hash("3"),
            receipt_id=_hash("4"),
            total_candidate_count=4,
            examined_candidate_count=4,
            accepted_labeled_count=accepted,
            rejected_labeled_count=rejected,
            representatives=(_representative("5", (0, 0, 0, 0)),),
            complete=True,
        )


def test_duplicate_cross_record_identifiers_and_empty_atlas_fail_closed() -> None:
    with pytest.raises(AtlasError, match="duplicate certificate"):
        AtlasRepresentative.create(
            canonical_id=_hash("1"),
            representative_outputs=(0,),
            full_orbit_size=1,
            accepted_member_count=1,
            certificate_ids=(_hash("2"), _hash("2")),
            analysis_ids=(_hash("3"),),
        )
    with pytest.raises(AtlasError, match="must contain a corpus"):
        FiniteAlgebraAtlas.create(())


def test_records_are_factory_owned_immutable_and_revalidated_before_encoding() -> None:
    representative = _representative("1", (0, 0, 1, 1))
    with pytest.raises(AtlasError, match="factory-owned"):
        AtlasRepresentative()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (AtlasRepresentative,), {})
    with pytest.raises(FrozenInstanceError):
        representative.full_orbit_size = 9  # type: ignore[misc]

    corpus = create_atlas_corpus(
        corpus_id="one-orbit",
        spec_id=_hash("2"),
        enumeration_id=_hash("3"),
        classification_id=_hash("4"),
        receipt_id=_hash("5"),
        total_candidate_count=2,
        examined_candidate_count=2,
        accepted_labeled_count=2,
        rejected_labeled_count=0,
        representatives=(representative,),
        complete=True,
    )
    atlas = FiniteAlgebraAtlas.create((corpus,))
    object.__setattr__(atlas.header, "representative_count", 99)
    with pytest.raises(AtlasError, match="content drift"):
        atlas_canonical_bytes(atlas)
