"""Tests for neutral exact sparse-linear utilities."""

from __future__ import annotations

from fractions import Fraction
import math
import sys

import pytest

from anyalgebra.experimental.exact_linear import (
    SparseRationalOperator,
    independent_sparse_operators,
    scaled_taylor_action,
    sparse_exponential_action,
    sparse_operator_coordinates,
    sparse_operator_max_norm,
)


def test_sparse_operator_composition_and_coordinates_are_exact() -> None:
    left = SparseRationalOperator.from_mapping(2, {(0, 1): 2, (1, 0): -1})
    right = SparseRationalOperator.from_mapping(2, {(0, 0): 1, (1, 1): 3})
    basis = (left, right)

    assert left @ right == SparseRationalOperator.from_mapping(
        2, {(0, 1): 6, (1, 0): -1}
    )
    assert left.commutator(right) == SparseRationalOperator.from_mapping(
        2, {(0, 1): 4, (1, 0): 2}
    )
    assert left.apply((Fraction(3, 2), Fraction(-1, 4))) == (
        Fraction(-1, 2),
        Fraction(-3, 2),
    )
    assert independent_sparse_operators((left, right, left.scale(2))) == basis
    assert sparse_operator_coordinates(left.scale(3).add(right.scale(-2)), basis) == (
        Fraction(3),
        Fraction(-2),
    )


def test_sparse_numerical_exponential_has_an_adaptive_inverse() -> None:
    rotation = SparseRationalOperator.from_mapping(2, {(0, 1): -1, (1, 0): 1})
    value = (1.0, 0.0)
    transformed = sparse_exponential_action(rotation, value, parameter=0.2)
    recovered = sparse_exponential_action(rotation, transformed, parameter=-0.2)

    error = max(abs(left - right) for left, right in zip(recovered, value, strict=True))
    assert error < 1e-13


def test_exact_sparse_operator_rejects_inexact_runtime_scalars() -> None:
    operator = SparseRationalOperator.from_mapping(2, {(0, 1): 1})

    with pytest.raises(TypeError, match="exact integer or Fraction"):
        SparseRationalOperator.from_mapping(2, {(0, 0): 0.5})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="exact integer or Fraction"):
        operator.scale(True)
    with pytest.raises(TypeError, match="exact integers or Fractions"):
        operator.apply((1, 0.5))  # type: ignore[arg-type]


def test_floating_sparse_operator_fails_closed_on_nonfinite_output() -> None:
    operator = SparseRationalOperator.from_mapping(1, {(0, 0): 2})

    with pytest.raises(ArithmeticError, match="became non-finite"):
        operator.apply_float((sys.float_info.max,))


def test_numerical_exponential_matches_closed_forms_at_large_parameters() -> None:
    rotation = SparseRationalOperator.from_mapping(2, {(0, 1): -1, (1, 0): 1})
    for parameter in (0.2, 1.0, 20.0, 40.0, 100.0, -37.5):
        cosine, sine = sparse_exponential_action(
            rotation, (1.0, 0.0), parameter=parameter
        )
        # A plain Taylor sum loses every digit near 40; this one may not.
        assert abs(cosine - math.cos(parameter)) < 1e-12
        assert abs(sine - math.sin(parameter)) < 1e-12

    decay = SparseRationalOperator.from_mapping(1, {(0, 0): -1})
    (value,) = sparse_exponential_action(decay, (1.0,), parameter=40.0)
    assert abs(value - math.exp(-40.0)) < 1e-12 * math.exp(-40.0)

    nilpotent = SparseRationalOperator.from_mapping(3, {(0, 1): 2, (1, 2): 3})
    assert sparse_exponential_action(nilpotent, (0.0, 0.0, 1.0)) == (3.0, 3.0, 1.0)
    assert sparse_exponential_action(rotation, (0.0, 0.0), parameter=5.0) == (0.0, 0.0)


def test_numerical_exponential_is_linear_at_every_scale() -> None:
    rotation = SparseRationalOperator.from_mapping(2, {(0, 1): -1, (1, 0): 1})
    reference = sparse_exponential_action(rotation, (1.0, 0.0), parameter=1.0)
    for scale in (1e-200, 1e-16, 1e-10, 1e30, 1e200):
        scaled = sparse_exponential_action(rotation, (scale, 0.0), parameter=1.0)
        assert all(
            abs(entry / scale - expected) < 1e-14
            for entry, expected in zip(scaled, reference, strict=True)
        )


def test_numerical_exponential_reports_what_it_cannot_do() -> None:
    rotation = SparseRationalOperator.from_mapping(2, {(0, 1): -1, (1, 0): 1})
    chain = SparseRationalOperator.from_mapping(
        3, {(1, 0): Fraction(1, 10**16), (2, 1): 10**18}
    )
    with pytest.raises(ArithmeticError, match="substeps"):
        sparse_exponential_action(chain, (1.0, 0.0, 0.0))
    with pytest.raises(ArithmeticError, match="did not converge within 3 terms"):
        sparse_exponential_action(rotation, (1.0, 0.0), parameter=3.0, max_terms=3)
    with pytest.raises(ValueError, match="finite"):
        sparse_exponential_action(rotation, (1.0, 0.0), parameter=float("nan"))
    with pytest.raises(TypeError, match="parameter"):
        sparse_exponential_action(rotation, (1.0, 0.0), parameter=True)
    with pytest.raises(TypeError, match="parameter"):
        sparse_exponential_action(
            rotation,
            (1.0, 0.0),
            parameter="0.5",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="tolerance"):
        sparse_exponential_action(rotation, (1.0, 0.0), tolerance=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        sparse_exponential_action(rotation, (1.0, 0.0), max_steps=0)
    with pytest.raises(ValueError, match="max_step_norm"):
        sparse_exponential_action(rotation, (1.0, 0.0), max_step_norm=0.0)
    with pytest.raises(TypeError, match="floats, integers, or Fractions"):
        sparse_exponential_action(rotation, (True, 0.0))
    with pytest.raises(ValueError, match="dimension"):
        sparse_exponential_action(rotation, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="norm bound"):
        scaled_taylor_action(lambda vector: vector, -1.0, (1.0,))
    with pytest.raises(ValueError, match="changed the vector dimension"):
        scaled_taylor_action(lambda vector: (*vector, 0.0), 1.0, (1.0,))


def test_operator_max_norm_is_the_largest_absolute_row_sum() -> None:
    operator = SparseRationalOperator.from_mapping(
        3, {(0, 0): 1, (0, 2): Fraction(-5, 2), (2, 1): 3}
    )
    assert sparse_operator_max_norm(operator) == 3.5
    assert sparse_operator_max_norm(SparseRationalOperator.from_mapping(2, {})) == 0.0


def test_constructor_stores_fractions_even_for_equal_integers() -> None:
    from_integers = SparseRationalOperator(2, ((0, 0, 2), (1, 1, 3)))  # type: ignore[arg-type]
    from_fractions = SparseRationalOperator.from_mapping(2, {(0, 0): 2, (1, 1): 3})

    assert all(type(value) is Fraction for _, _, value in from_integers.entries)
    assert from_integers == from_fractions
    assert hash(from_integers) == hash(from_fractions)
    # Integer storage used to make the reducer divide in floating point.
    assert len(independent_sparse_operators((from_integers, from_integers))) == 1
    assert sparse_operator_coordinates(from_integers, (from_integers,)) == (
        Fraction(1),
    )

    with pytest.raises(TypeError, match="exact integer or Fraction"):
        SparseRationalOperator(1, ((0, 0, True),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact integer or Fraction"):
        SparseRationalOperator(1, ((0, 0, 0.5),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical"):
        SparseRationalOperator(2, ((1, 1, Fraction(1)), (0, 0, Fraction(1))))
    with pytest.raises(ValueError, match="canonical"):
        SparseRationalOperator(1, ((0, 0, Fraction(0)),))


def test_operator_hash_is_value_based_and_arithmetic_stays_canonical() -> None:
    left = SparseRationalOperator.from_mapping(2, {(0, 1): 2, (1, 0): -1})
    right = SparseRationalOperator.from_mapping(2, {(0, 0): 1, (1, 1): 3})
    assert hash(left) == hash((left.dimension, left.entries))
    assert hash(left) == hash(left)

    for result in (
        left.add(right),
        left.add(left.scale(-1)),
        left.scale(Fraction(1, 3)),
        left.scale(0),
        left.transpose(),
        left @ right,
        left.commutator(left),
    ):
        # Re-validating through the public constructor must not change anything.
        assert SparseRationalOperator(result.dimension, result.entries) == result
        assert all(type(value) is Fraction for _, _, value in result.entries)
    assert left.add(left.scale(-1)).entries == ()
    assert left.transpose().entries == ((0, 1, Fraction(-1)), (1, 0, Fraction(2)))

    with pytest.raises(TypeError):
        left @ 3  # type: ignore[operator]
    with pytest.raises(TypeError, match="sparse operators"):
        left.add(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floats, integers, or Fractions"):
        left.apply_float((True, 0.0))
