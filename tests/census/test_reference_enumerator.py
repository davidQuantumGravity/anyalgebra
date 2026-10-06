"""Contracts for the simple unpruned reference enumerator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.reference import (
    ReferenceEnumerationError,
    enumerate_reference,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import rank_operation_table


def _spec(
    size: int,
    arity: int,
    *,
    max_candidates: int = 100_000,
    max_work_units: int = 100_000,
    max_memory_bytes: int = 16_000_000,
    max_orbits: int = 100_000,
    observed: int | None = None,
    constrained: bool = False,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"reference-{size}-{arity}",
    )
    constraints = (
        (CensusConstraint.commutative(),) if constrained and arity == 2 else ()
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=max_orbits,
            max_work_units=max_work_units,
            max_memory_bytes=max_memory_bytes,
            max_observed_milliseconds=observed,
        ),
    )


@pytest.mark.parametrize(
    ("size", "arity", "expected"),
    (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 3, 1),
        (2, 0, 2),
        (2, 1, 4),
        (2, 2, 16),
        (3, 0, 3),
        (3, 1, 27),
    ),
)
def test_empty_singleton_unary_and_binary_counts_are_exact(
    size: int, arity: int, expected: int
) -> None:
    result = enumerate_reference(_spec(size, arity))

    assert result.complete is True
    assert result.total_candidate_count == expected
    assert result.examined_count == expected
    assert result.emitted_count == expected
    assert len(result.candidates) == expected
    assert result.next_candidate_index is None
    assert result.stop_reason is None


def test_candidates_are_emitted_in_canonical_index_order() -> None:
    result = enumerate_reference(_spec(2, 2))

    assert tuple(candidate.candidate_index for candidate in result.candidates) == tuple(
        range(16)
    )
    assert tuple(candidate.outputs for candidate in result.candidates[:5]) == (
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 0),
    )
    assert all(
        rank_operation_table(result.core, candidate.outputs)
        == candidate.candidate_index
        for candidate in result.candidates
    )


def test_candidate_bound_returns_exact_incomplete_frontier() -> None:
    result = enumerate_reference(_spec(3, 2, max_candidates=5))

    assert result.complete is False
    assert result.examined_count == result.emitted_count == 5
    assert result.next_candidate_index == 5
    assert result.stop_reason == "max_candidates"
    assert result.candidates[-1].candidate_index == 4


def test_work_bound_returns_exact_incomplete_frontier() -> None:
    result = enumerate_reference(_spec(3, 2, max_candidates=10, max_work_units=3))

    assert result.complete is False
    assert result.examined_count == result.emitted_count == 3
    assert result.next_candidate_index == 3
    assert result.stop_reason == "max_work_units"


def test_memory_bound_is_charged_before_candidate_generation() -> None:
    result = enumerate_reference(_spec(2, 2, max_memory_bytes=191))

    assert result.workspace_bytes_per_candidate == 192
    assert result.candidates == ()
    assert result.examined_count == 0
    assert result.next_candidate_index == 0
    assert result.stop_reason == "max_memory_bytes"


def test_zero_orbit_and_observation_bounds_do_not_change_raw_enumeration() -> None:
    result = enumerate_reference(_spec(2, 1, max_orbits=0, observed=0))

    assert result.complete is True
    assert result.emitted_count == 4


def test_constraints_are_not_silently_applied_before_filtering_task() -> None:
    result = enumerate_reference(_spec(2, 2, constrained=True))

    assert result.total_candidate_count == 16
    assert result.emitted_count == 16


def test_tied_bounds_use_canonical_candidate_work_memory_order() -> None:
    candidate_first = enumerate_reference(
        _spec(2, 2, max_candidates=0, max_work_units=0, max_memory_bytes=0)
    )
    work_second = enumerate_reference(
        _spec(2, 2, max_candidates=1, max_work_units=0, max_memory_bytes=0)
    )

    assert candidate_first.stop_reason == "max_candidates"
    assert work_second.stop_reason == "max_work_units"


def test_records_are_factory_owned_sealed_and_immutable() -> None:
    result = enumerate_reference(_spec(2, 1))
    candidate = result.candidates[0]

    with pytest.raises(ReferenceEnumerationError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.complete = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_index = 2  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_invalid_spec_fails_before_any_candidate_generation() -> None:
    with pytest.raises(ReferenceEnumerationError) as caught:
        enumerate_reference(cast(CensusSpec, None))
    assert caught.value.field == "spec"


def test_large_corpus_with_zero_budget_returns_without_dense_allocation() -> None:
    result = enumerate_reference(
        _spec(2, 16, max_candidates=0, max_work_units=0, max_memory_bytes=0)
    )

    assert result.total_candidate_count.bit_length() == 65_537
    assert result.candidates == ()
    assert result.next_candidate_index == 0
    assert result.complete is False
