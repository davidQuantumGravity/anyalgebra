"""Contracts for allow-listed declarative finite-census constraints."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from anyalgebra.census.constraints import (
    CensusConstraint,
    CensusTerm,
    ConstraintError,
    ConstraintSet,
)
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int = 3, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"control-{size}-{arity}",
    )


def test_allow_list_canonicalizes_order_and_exact_duplicates() -> None:
    constraints = (
        CensusConstraint.quasigroup(),
        CensusConstraint.identity(element=0, side="two_sided"),
        CensusConstraint.commutative(),
        CensusConstraint.idempotent(),
        CensusConstraint.commutative(),
    )
    left = ConstraintSet.create(_core(), constraints)
    right = ConstraintSet.create(_core(), reversed(constraints))

    assert left == right
    assert tuple(item.kind for item in left.constraints) == (
        "commutative",
        "idempotent",
        "identity",
        "quasigroup",
    )
    assert (
        repr(left) == "ConstraintSet(count=4, kinds=("
        "'commutative', 'idempotent', 'identity', 'quasigroup'))"
    )
    assert not hasattr(left, "__dict__")
    with pytest.raises(FrozenInstanceError):
        left.constraints = ()  # type: ignore[misc]


@pytest.mark.parametrize("side", ("left", "right", "two_sided"))
def test_identity_sides_and_carrier_indices_are_explicit(side: str) -> None:
    constraint = CensusConstraint.identity(element=2, side=side)
    result = ConstraintSet.create(_core(), (constraint,))

    assert result.constraints[0].parameters == (("element", 2), ("side", side))


def test_nullary_value_is_only_for_nullary_operation_and_in_range() -> None:
    constraint = CensusConstraint.nullary_value(element=1)
    assert ConstraintSet.create(_core(2, 0), (constraint,)).constraints == (constraint,)

    with pytest.raises(ConstraintError, match="nullary"):
        ConstraintSet.create(_core(2, 1), (constraint,))
    with pytest.raises(ConstraintError, match="carrier range"):
        ConstraintSet.create(_core(2, 0), (CensusConstraint.nullary_value(element=2),))


@pytest.mark.parametrize(
    "constraint",
    (
        CensusConstraint.commutative(),
        CensusConstraint.idempotent(),
        CensusConstraint.quasigroup(),
        CensusConstraint.identity(element=0, side="left"),
    ),
)
def test_binary_constraints_reject_nonbinary_operation(
    constraint: CensusConstraint,
) -> None:
    with pytest.raises(ConstraintError, match="binary"):
        ConstraintSet.create(_core(2, 3), (constraint,))


def test_equation_terms_are_declarative_canonical_and_arity_checked() -> None:
    x = CensusTerm.variable(0)
    y = CensusTerm.variable(1)
    xy = CensusTerm.apply(x, y)
    yx = CensusTerm.apply(y, x)
    equation = CensusConstraint.equation(xy, yx, variable_count=2)
    result = ConstraintSet.create(_core(), (equation,))

    assert result.constraints == (equation,)
    assert equation.kind == "equation"
    assert equation.parameters == (("variable_count", 2),)
    assert equation.left == xy
    assert equation.right == yx

    malformed = CensusConstraint.equation(CensusTerm.apply(x), x, variable_count=1)
    with pytest.raises(ConstraintError, match="operation arity"):
        ConstraintSet.create(_core(), (malformed,))


def test_equation_rejects_out_of_range_variables_and_constants() -> None:
    with pytest.raises(ConstraintError, match="variable range"):
        ConstraintSet.create(
            _core(),
            (
                CensusConstraint.equation(
                    CensusTerm.variable(1), CensusTerm.variable(0), variable_count=1
                ),
            ),
        )
    with pytest.raises(ConstraintError, match="carrier range"):
        ConstraintSet.create(
            _core(2),
            (
                CensusConstraint.equation(
                    CensusTerm.constant(2), CensusTerm.constant(0), variable_count=0
                ),
            ),
        )


def test_callable_unknown_and_direct_construction_are_rejected() -> None:
    with pytest.raises(ConstraintError, match="exact CensusConstraint"):
        ConstraintSet.create(_core(), (lambda table: True,))  # type: ignore[arg-type]
    with pytest.raises(ConstraintError, match="factory-owned"):
        CensusConstraint()
    with pytest.raises(ConstraintError, match="factory-owned"):
        CensusTerm()
    with pytest.raises(ConstraintError, match="factory-owned"):
        ConstraintSet()

    with pytest.raises(ConstraintError, match="side"):
        CensusConstraint.identity(element=0, side="unknown")


def test_sources_are_snapshotted_once_and_bounded_before_user_work() -> None:
    source = [CensusConstraint.commutative()]
    result = ConstraintSet.create(_core(), (item for item in source))
    source.clear()
    assert len(result.constraints) == 1

    with pytest.raises(ConstraintError, match="declaration limit"):
        ConstraintSet.create(
            _core(), (CensusConstraint.commutative() for _ in range(257))
        )


def test_equation_terms_have_hard_arity_depth_and_node_preflights() -> None:
    with pytest.raises(ConstraintError, match="arity limit"):
        CensusTerm.apply(*(CensusTerm.variable(0) for _ in range(65)))

    term = CensusTerm.variable(0)
    for _ in range(63):
        term = CensusTerm.apply(term)
    with pytest.raises(ConstraintError, match="depth limit"):
        CensusTerm.apply(term)

    leaves = tuple(CensusTerm.variable(0) for _ in range(64))
    level = CensusTerm.apply(*leaves)
    for _ in range(5):
        level = CensusTerm.apply(*(level for _ in range(2)))
    with pytest.raises(ConstraintError, match="node limit"):
        CensusTerm.apply(*(level for _ in range(64)))


def test_records_are_unhashable_and_subclass_sealed() -> None:
    values = (
        CensusTerm.variable(0),
        CensusConstraint.commutative(),
        ConstraintSet.create(_core(), ()),
    )
    for value in values:
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class Attempt(CensusConstraint):
            pass
