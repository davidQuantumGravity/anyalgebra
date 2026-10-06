"""Contracts for deterministic evaluation of declarative census constraints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from itertools import product
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import (
    CensusConstraint,
    CensusTerm,
    ConstraintSet,
)
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import (
    ConstraintFilterError,
    compile_constraint_filter,
)
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpecCore


def _spec(
    size: int,
    constraints: tuple[CensusConstraint, ...],
    *,
    arity: int = 2,
    work: int = 1_000_000,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"filter-{size}-{arity}",
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100_000,
            max_orbits=100_000,
            max_work_units=work,
            max_memory_bytes=100_000_000,
        ),
    )


def _accepted_indices(spec: CensusSpec) -> tuple[int, ...]:
    compiled = compile_constraint_filter(spec)
    raw = enumerate_reference(spec)
    return tuple(
        candidate.candidate_index
        for candidate in raw.candidates
        if compiled.evaluate(candidate).accepted
    )


def _direct_binary(outputs: tuple[int, ...], size: int, left: int, right: int) -> int:
    return outputs[left * size + right]


@pytest.mark.parametrize(
    ("constraint", "direct"),
    (
        (
            CensusConstraint.commutative(),
            lambda table: all(
                _direct_binary(table, 2, x, y) == _direct_binary(table, 2, y, x)
                for x, y in product(range(2), repeat=2)
            ),
        ),
        (
            CensusConstraint.idempotent(),
            lambda table: all(_direct_binary(table, 2, x, x) == x for x in range(2)),
        ),
        (
            CensusConstraint.identity(element=0, side="two_sided"),
            lambda table: all(
                _direct_binary(table, 2, 0, x) == x
                and _direct_binary(table, 2, x, 0) == x
                for x in range(2)
            ),
        ),
        (
            CensusConstraint.quasigroup(),
            lambda table: all(
                {_direct_binary(table, 2, x, y) for y in range(2)} == {0, 1}
                and {_direct_binary(table, 2, y, x) for y in range(2)} == {0, 1}
                for x in range(2)
            ),
        ),
    ),
)
def test_builtin_filter_agrees_with_independent_direct_predicate(
    constraint: CensusConstraint, direct: Callable[[tuple[int, ...]], bool]
) -> None:
    spec = _spec(2, (constraint,))
    compiled = compile_constraint_filter(spec)
    raw = enumerate_reference(spec)

    observed = tuple(
        candidate.candidate_index
        for candidate in raw.candidates
        if compiled.evaluate(candidate).accepted
    )
    expected = tuple(
        candidate.candidate_index
        for candidate in raw.candidates
        if direct(candidate.outputs)
    )
    assert observed == expected


def test_hand_counts_for_binary_order_two_constraints() -> None:
    assert len(_accepted_indices(_spec(2, (CensusConstraint.commutative(),)))) == 8
    assert len(_accepted_indices(_spec(2, (CensusConstraint.idempotent(),)))) == 4
    assert (
        len(
            _accepted_indices(
                _spec(
                    2,
                    (
                        CensusConstraint.commutative(),
                        CensusConstraint.idempotent(),
                    ),
                )
            )
        )
        == 2
    )
    assert (
        len(
            _accepted_indices(
                _spec(
                    2,
                    (CensusConstraint.identity(element=0, side="two_sided"),),
                )
            )
        )
        == 2
    )
    assert len(_accepted_indices(_spec(2, (CensusConstraint.quasigroup(),)))) == 2


def test_equation_filter_agrees_with_direct_associativity() -> None:
    x, y, z = (CensusTerm.variable(index) for index in range(3))
    equation = CensusConstraint.equation(
        CensusTerm.apply(CensusTerm.apply(x, y), z),
        CensusTerm.apply(x, CensusTerm.apply(y, z)),
        variable_count=3,
    )
    spec = _spec(2, (equation,))
    raw = enumerate_reference(spec)
    compiled = compile_constraint_filter(spec)

    observed = tuple(
        item.candidate_index
        for item in raw.candidates
        if compiled.evaluate(item).accepted
    )
    expected = tuple(
        item.candidate_index
        for item in raw.candidates
        if all(
            _direct_binary(
                item.outputs,
                2,
                _direct_binary(item.outputs, 2, a, b),
                c,
            )
            == _direct_binary(
                item.outputs,
                2,
                a,
                _direct_binary(item.outputs, 2, b, c),
            )
            for a, b, c in product(range(2), repeat=3)
        )
    )
    assert observed == expected
    assert len(observed) == 8


def test_nullary_value_filter_and_empty_assignment_boundary() -> None:
    nullary = _spec(2, (CensusConstraint.nullary_value(element=1),), arity=0)
    result = compile_constraint_filter(nullary)
    raw = enumerate_reference(nullary)
    assert tuple(result.evaluate(item).accepted for item in raw.candidates) == (
        False,
        True,
    )

    x = CensusTerm.variable(0)
    empty = _spec(
        0,
        (CensusConstraint.equation(x, x, variable_count=1),),
        arity=1,
    )
    empty_candidate = enumerate_reference(empty).candidates[0]
    check = compile_constraint_filter(empty).evaluate(empty_candidate).checks[0]
    assert check.accepted is True
    assert check.checked_cases == 0


def test_all_constraints_are_evaluated_after_an_earlier_rejection() -> None:
    spec = _spec(
        2,
        (
            CensusConstraint.commutative(),
            CensusConstraint.idempotent(),
            CensusConstraint.identity(element=0, side="two_sided"),
            CensusConstraint.quasigroup(),
        ),
    )
    candidate = enumerate_reference(spec).candidates[0]
    result = compile_constraint_filter(spec).evaluate(candidate)

    assert result.accepted is False
    assert len(result.checks) == 4
    assert all(check.checked_cases > 0 for check in result.checks)
    assert result.rejection_reasons == tuple(
        check.reason for check in result.checks if not check.accepted
    )


def test_reordered_constraints_have_identical_checks_and_rejection_summary() -> None:
    declarations = (
        CensusConstraint.idempotent(),
        CensusConstraint.commutative(),
        CensusConstraint.identity(element=0, side="left"),
    )
    first = _spec(2, declarations)
    second = _spec(2, tuple(reversed(declarations)))
    first_candidate = enumerate_reference(first).candidates[5]
    second_candidate = enumerate_reference(second).candidates[5]

    left = compile_constraint_filter(first).evaluate(first_candidate)
    right = compile_constraint_filter(second).evaluate(second_candidate)
    assert left == right


def test_failed_checks_have_counts_and_first_canonical_witnesses() -> None:
    spec = _spec(2, (CensusConstraint.commutative(),))
    candidate = enumerate_reference(spec).candidates[2]
    check = compile_constraint_filter(spec).evaluate(candidate).checks[0]

    assert check.accepted is False
    assert check.reason == "commutativity_mismatch"
    assert check.checked_cases == 4
    assert check.violation_count == 2
    assert check.first_witness == (0, 1)


def test_work_is_preflighted_when_filter_is_compiled() -> None:
    spec = _spec(3, (CensusConstraint.commutative(),), work=8)

    with pytest.raises(ConstraintFilterError) as caught:
        compile_constraint_filter(spec)
    assert caught.value.field == "max_work_units"
    assert caught.value.required_work == 9


def test_candidate_from_incompatible_core_fails_closed() -> None:
    binary = _spec(2, (CensusConstraint.commutative(),))
    ternary_carrier = _spec(3, ())
    candidate = enumerate_reference(ternary_carrier).candidates[0]

    with pytest.raises(ConstraintFilterError) as caught:
        compile_constraint_filter(binary).evaluate(candidate)
    assert caught.value.field == "candidate"


def test_compiled_filter_and_results_are_immutable_factory_owned_records() -> None:
    compiled = compile_constraint_filter(_spec(2, (CensusConstraint.idempotent(),)))
    candidate = enumerate_reference(compiled.spec).candidates[0]
    result = compiled.evaluate(candidate)

    with pytest.raises(ConstraintFilterError, match="factory-owned"):
        type(compiled)()
    with pytest.raises(FrozenInstanceError):
        result.accepted = True  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    assert not hasattr(result, "__dict__")


def test_compile_rejects_nonexact_spec_without_consuming_candidate_data() -> None:
    with pytest.raises(ConstraintFilterError) as caught:
        compile_constraint_filter(cast(CensusSpec, None))
    assert caught.value.field == "spec"
