"""Exact canonical-orbit classification of complete accepted streams."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.classify import (
    OrbitClassificationError,
    classify_reference_orbits,
    orbit_classification_canonical_bytes,
)
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.reference import ReferenceEnumeration, enumerate_reference
from anyalgebra.census.spec import CensusSpecCore


def _spec(
    size: int,
    arity: int = 2,
    *,
    constraints: tuple[CensusConstraint, ...] = (),
    literal: bool = False,
    max_candidates: int = 100_000,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"orbit-classification-{size}-{arity}-{literal}",
    )
    policy = (
        EquivalencePolicy.literal(core)
        if literal
        else EquivalencePolicy.relabeling(core)
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=policy,
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=100_000,
            max_work_units=1_000_000,
            max_memory_bytes=100_000_000,
        ),
    )


def test_all_order_two_binary_tables_form_ten_exact_orbits() -> None:
    spec = _spec(2)
    result = classify_reference_orbits(enumerate_reference(spec))

    assert result.complete is True
    assert result.raw_labeled_count == 16
    assert result.accepted_labeled_count == 16
    assert result.rejected_labeled_count == 0
    assert result.orbit_count == 10
    assert sum(orbit.accepted_member_count for orbit in result.orbits) == 16
    assert sorted(orbit.accepted_member_count for orbit in result.orbits) == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
    ]


def test_every_member_certificate_replays_and_every_representative_is_canonical() -> (
    None
):
    spec = _spec(2)
    result = classify_reference_orbits(enumerate_reference(spec))
    seen_indices: list[int] = []
    for orbit in result.orbits:
        representative = canonicalize_operation_table(
            spec.core, orbit.representative_outputs, spec.equivalence
        )
        assert representative.canonical_outputs == orbit.representative_outputs
        assert representative.canonical_id == orbit.canonical_id
        assert orbit.full_orbit_size == representative.orbit_size
        for member in orbit.members:
            seen_indices.append(member.candidate_index)
            assert member.certificate.canonical_id == orbit.canonical_id
            assert member.verification.transport_verified is True
            assert member.verification.identifier_verified is True
            assert member.verification.reconstructed_source == member.outputs
            assert member.verification.reconstructed_target == (
                orbit.representative_outputs
            )

    assert sorted(seen_indices) == list(range(16))


def test_constraints_are_reapplied_before_orbit_membership() -> None:
    spec = _spec(2, constraints=(CensusConstraint.commutative(),))
    result = classify_reference_orbits(enumerate_reference(spec))

    assert result.raw_labeled_count == 16
    assert result.accepted_labeled_count == 8
    assert result.rejected_labeled_count == 8
    assert sum(orbit.accepted_member_count for orbit in result.orbits) == 8
    assert all(
        member.outputs[1] == member.outputs[2]
        for orbit in result.orbits
        for member in orbit.members
    )


def test_literal_equivalence_produces_one_orbit_per_accepted_table() -> None:
    spec = _spec(2, literal=True)
    result = classify_reference_orbits(enumerate_reference(spec))

    assert result.orbit_count == result.accepted_labeled_count == 16
    assert all(
        orbit.accepted_member_count == orbit.full_orbit_size == 1
        for orbit in result.orbits
    )


def test_incomplete_reference_stream_cannot_serialize_as_classified() -> None:
    spec = _spec(2, max_candidates=3)
    enumeration = enumerate_reference(spec)
    assert enumeration.complete is False
    with pytest.raises(OrbitClassificationError) as caught:
        classify_reference_orbits(enumeration)
    assert caught.value.field == "enumeration"


def test_canonical_bytes_are_deterministic_content_addressed_and_sorted() -> None:
    spec = _spec(2)
    first_result = classify_reference_orbits(enumerate_reference(spec))
    second_result = classify_reference_orbits(enumerate_reference(spec))
    first = orbit_classification_canonical_bytes(first_result)
    second = orbit_classification_canonical_bytes(second_result)
    record = json.loads(first)

    assert first_result == second_result
    assert first == second
    assert [item["canonicalId"]["digest"] for item in record["orbits"]] == sorted(
        item["canonicalId"]["digest"] for item in record["orbits"]
    )
    assert record["contentHash"]["digest"] == first_result.semantic_hash.digest


def test_nullary_and_empty_carrier_boundaries_classify() -> None:
    nullary = classify_reference_orbits(enumerate_reference(_spec(2, 0)))
    assert nullary.accepted_labeled_count == 2
    assert nullary.orbit_count == 1
    assert nullary.orbits[0].accepted_member_count == 2

    empty = classify_reference_orbits(enumerate_reference(_spec(0, 1)))
    assert empty.accepted_labeled_count == empty.orbit_count == 1
    assert empty.orbits[0].representative_outputs == ()


def test_nonexact_input_and_result_mutation_fail_closed() -> None:
    with pytest.raises(OrbitClassificationError) as caught:
        classify_reference_orbits(cast(ReferenceEnumeration, None))
    assert caught.value.field == "enumeration"

    result = classify_reference_orbits(enumerate_reference(_spec(2)))
    with pytest.raises(OrbitClassificationError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.orbit_count = 0  # type: ignore[misc]

    object.__setattr__(result, "accepted_labeled_count", 0)
    with pytest.raises(OrbitClassificationError):
        orbit_classification_canonical_bytes(result)
