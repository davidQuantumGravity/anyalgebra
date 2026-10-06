"""Exact finite-magma nuclei, commutant, and center controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product
import json

import pytest

from anyalgebra.census.analysis import (
    FiniteLawAnalysisError,
    analyze_finite_magma_subobjects,
    finite_magma_subobject_canonical_bytes,
)
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"finite-subobjects-{size}-{arity}",
    )


def _op(outputs: tuple[int, ...], size: int, left: int, right: int) -> int:
    return outputs[left * size + right]


def test_associative_commutative_control_has_full_carrier_center() -> None:
    core = _core(2)
    result = analyze_finite_magma_subobjects(core, (0, 1, 1, 0))

    expected = (0, 1)
    assert result.left_nucleus == expected
    assert result.middle_nucleus == expected
    assert result.right_nucleus == expected
    assert result.nucleus == expected
    assert result.commutant == expected
    assert result.center == expected
    assert result.status == "computed_complete"
    assert result.complete is True
    assert result.associator_assignments_checked == 3 * core.carrier_size**3
    assert result.commutator_pairs_checked == core.carrier_size**2


def test_nonassociative_control_distinguishes_all_three_slots() -> None:
    core = _core(2)
    outputs = (0, 0, 1, 0)
    result = analyze_finite_magma_subobjects(core, outputs)

    assert result.left_nucleus == (0,)
    assert result.middle_nucleus == ()
    assert result.right_nucleus == (0,)
    assert result.nucleus == ()
    assert result.commutant == ()
    assert result.center == ()

    for candidate in result.left_nucleus:
        assert all(
            _op(outputs, 2, _op(outputs, 2, candidate, x), y)
            == _op(outputs, 2, candidate, _op(outputs, 2, x, y))
            for x in range(2)
            for y in range(2)
        )
    for candidate in result.right_nucleus:
        assert all(
            _op(outputs, 2, _op(outputs, 2, x, y), candidate)
            == _op(outputs, 2, x, _op(outputs, 2, y, candidate))
            for x in range(2)
            for y in range(2)
        )


def test_center_is_exactly_nucleus_intersect_commutant() -> None:
    core = _core(2)
    outputs = (0, 0, 0, 1)
    result = analyze_finite_magma_subobjects(core, outputs)

    assert result.nucleus == (0, 1)
    assert result.commutant == (0, 1)
    assert result.center == tuple(
        item for item in result.nucleus if item in result.commutant
    )


def test_every_order_two_table_agrees_with_independent_definitions() -> None:
    core = _core(2)
    carrier = range(2)
    for outputs in product(carrier, repeat=4):
        result = analyze_finite_magma_subobjects(core, outputs)
        left = tuple(
            a
            for a in carrier
            if all(
                _op(outputs, 2, _op(outputs, 2, a, x), y)
                == _op(outputs, 2, a, _op(outputs, 2, x, y))
                for x in carrier
                for y in carrier
            )
        )
        middle = tuple(
            a
            for a in carrier
            if all(
                _op(outputs, 2, _op(outputs, 2, x, a), y)
                == _op(outputs, 2, x, _op(outputs, 2, a, y))
                for x in carrier
                for y in carrier
            )
        )
        right = tuple(
            a
            for a in carrier
            if all(
                _op(outputs, 2, _op(outputs, 2, x, y), a)
                == _op(outputs, 2, x, _op(outputs, 2, y, a))
                for x in carrier
                for y in carrier
            )
        )
        commutant = tuple(
            a
            for a in carrier
            if all(_op(outputs, 2, a, x) == _op(outputs, 2, x, a) for x in carrier)
        )
        nucleus = tuple(a for a in carrier if a in left and a in middle and a in right)
        assert result.left_nucleus == left
        assert result.middle_nucleus == middle
        assert result.right_nucleus == right
        assert result.nucleus == nucleus
        assert result.commutant == commutant
        assert result.center == tuple(a for a in nucleus if a in commutant)


def test_nonbinary_operations_are_explicitly_unsupported() -> None:
    core = _core(3, 1)
    result = analyze_finite_magma_subobjects(core, (1, 2, 0))

    assert result.status == "unsupported_nonbinary"
    assert result.complete is False
    assert result.left_nucleus == result.center == ()
    assert result.associator_assignments_checked == 0
    assert result.commutator_pairs_checked == 0


def test_empty_binary_carrier_is_a_complete_vacuous_boundary() -> None:
    result = analyze_finite_magma_subobjects(_core(0), ())

    assert result.status == "computed_complete"
    assert result.complete is True
    assert result.left_nucleus == result.commutant == result.center == ()
    assert result.associator_assignments_checked == 0
    assert result.commutator_pairs_checked == 0


def test_invalid_core_and_table_fail_closed() -> None:
    core = _core(2)
    with pytest.raises(FiniteLawAnalysisError) as caught:
        analyze_finite_magma_subobjects(None, ())  # type: ignore[arg-type]
    assert caught.value.field == "core"
    with pytest.raises(FiniteLawAnalysisError) as caught:
        analyze_finite_magma_subobjects(core, (0, 0, 0))
    assert caught.value.field == "outputs"


def test_result_is_sealed_deterministic_and_replayed_before_serialization() -> None:
    core = _core(2)
    outputs = (0, 0, 1, 0)
    first_result = analyze_finite_magma_subobjects(core, outputs)
    second_result = analyze_finite_magma_subobjects(core, outputs)
    first = finite_magma_subobject_canonical_bytes(first_result)

    assert first_result == second_result
    assert first == finite_magma_subobject_canonical_bytes(second_result)
    assert (
        json.loads(first)["contentHash"]["digest"] == first_result.semantic_hash.digest
    )
    with pytest.raises(FiniteLawAnalysisError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first_result),), {})
    with pytest.raises(FrozenInstanceError):
        first_result.complete = False  # type: ignore[misc]
    object.__setattr__(first_result, "center", (0,))
    with pytest.raises(FiniteLawAnalysisError, match="content drift"):
        finite_magma_subobject_canonical_bytes(first_result)
