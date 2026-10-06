"""Contracts for census resource ceilings and canonical ordering metadata."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.census.bounds import (
    CensusBoundExceeded,
    CensusBoundsError,
    CensusOrdering,
    EnumerationBounds,
)


ROOT = Path(__file__).resolve().parents[2]


def _bounds() -> EnumerationBounds:
    return EnumerationBounds.create(
        max_candidates=16,
        max_orbits=10,
        max_work_units=1_000,
        max_memory_bytes=1_048_576,
        max_observed_milliseconds=5_000,
    )


def test_bounds_are_exact_immutable_and_include_separate_observation_limit() -> None:
    bounds = _bounds()

    assert bounds.max_candidates == 16
    assert bounds.max_orbits == 10
    assert bounds.max_work_units == 1_000
    assert bounds.max_memory_bytes == 1_048_576
    assert bounds.max_observed_milliseconds == 5_000
    assert repr(bounds) == (
        "EnumerationBounds(max_candidates=16, max_orbits=10, "
        "max_work_units=1000, max_memory_bytes=1048576, "
        "max_observed_milliseconds=5000)"
    )
    assert not hasattr(bounds, "__dict__")
    with pytest.raises(FrozenInstanceError):
        bounds.max_candidates = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(bounds)


def test_zero_hard_ceilings_are_valid_and_checked_before_work() -> None:
    bounds = EnumerationBounds.create(
        max_candidates=0,
        max_orbits=0,
        max_work_units=0,
        max_memory_bytes=0,
    )

    bounds.preflight(candidate_count=0, orbit_count=0, work_units=0, memory_bytes=0)
    with pytest.raises(CensusBoundExceeded) as caught:
        bounds.preflight(candidate_count=1, orbit_count=0, work_units=0, memory_bytes=0)
    assert caught.value.field == "candidate_count"
    assert caught.value.limit == 0
    assert caught.value.requested == 1


@pytest.mark.parametrize(
    "field",
    ("max_candidates", "max_orbits", "max_work_units", "max_memory_bytes"),
)
@pytest.mark.parametrize("value", (True, -1, 1.5, "1", None))
def test_hard_ceiling_types_reject_bool_negative_and_noninteger(
    field: str, value: object
) -> None:
    max_candidates = cast(int, value) if field == "max_candidates" else 1
    max_orbits = cast(int, value) if field == "max_orbits" else 1
    max_work_units = cast(int, value) if field == "max_work_units" else 1
    max_memory_bytes = cast(int, value) if field == "max_memory_bytes" else 1

    with pytest.raises(CensusBoundsError) as caught:
        EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=max_orbits,
            max_work_units=max_work_units,
            max_memory_bytes=max_memory_bytes,
        )
    assert caught.value.field == field


@pytest.mark.parametrize("value", (True, -1, 1.5, "1"))
def test_observation_limit_is_none_or_nonnegative_integer(value: object) -> None:
    with pytest.raises(CensusBoundsError) as caught:
        EnumerationBounds.create(
            max_candidates=1,
            max_orbits=1,
            max_work_units=1,
            max_memory_bytes=1,
            max_observed_milliseconds=cast(int, value),
        )
    assert caught.value.field == "max_observed_milliseconds"

    assert (
        EnumerationBounds.create(
            max_candidates=1,
            max_orbits=1,
            max_work_units=1,
            max_memory_bytes=1,
            max_observed_milliseconds=None,
        ).max_observed_milliseconds
        is None
    )


def test_preflight_diagnostics_use_fixed_field_order_and_exact_witnesses() -> None:
    bounds = _bounds()

    with pytest.raises(CensusBoundExceeded) as caught:
        bounds.preflight(
            candidate_count=17,
            orbit_count=11,
            work_units=1_001,
            memory_bytes=1_048_577,
        )
    assert caught.value.field == "candidate_count"
    assert str(caught.value) == (
        "census preflight candidate_count exceeds declared limit 16: requested 17"
    )

    with pytest.raises(CensusBoundExceeded) as first_field:
        bounds.preflight(
            candidate_count=17,
            orbit_count=-1,
            work_units=-1,
            memory_bytes=-1,
        )
    assert first_field.value.field == "candidate_count"

    with pytest.raises(CensusBoundsError, match="non-negative"):
        bounds.preflight(
            candidate_count=0, orbit_count=-1, work_units=0, memory_bytes=0
        )

    with pytest.raises(CensusBoundsError, match="hard integer limit"):
        EnumerationBounds.create(
            max_candidates=1 << 63,
            max_orbits=0,
            max_work_units=0,
            max_memory_bytes=0,
        )


def test_reference_ordering_freezes_every_combinatorial_axis() -> None:
    ordering = CensusOrdering.reference()

    assert ordering.candidate_order == "lexicographic_flat_outputs"
    assert ordering.input_tuple_order == "lexicographic_indices"
    assert ordering.constraint_order == "canonical_constraint_key"
    assert ordering.permutation_order == "lexicographic_images"
    assert ordering.output_digit_significance == "first_cell_most_significant"
    assert ordering.schema_version == 1
    assert ordering == CensusOrdering.reference()
    assert repr(ordering) == "CensusOrdering(reference-v1)"


def test_ordering_and_bounds_are_factory_owned_sealed_and_unhashable() -> None:
    with pytest.raises(CensusBoundsError, match="factory-owned"):
        EnumerationBounds()
    with pytest.raises(CensusBoundsError, match="factory-owned"):
        CensusOrdering()
    for value in (_bounds(), CensusOrdering.reference()):
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class Attempt(CensusOrdering):
            pass


def test_ordering_metadata_is_hash_seed_and_platform_independent() -> None:
    script = (
        "from anyalgebra.census.bounds import CensusOrdering; "
        "o=CensusOrdering.reference(); "
        "print('|'.join((o.candidate_order,o.input_tuple_order,"
        "o.constraint_order,o.permutation_order,o.output_digit_significance)))"
    )
    outputs = []
    for seed in ("1", "923"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
    assert outputs[0].endswith("first_cell_most_significant\n")
