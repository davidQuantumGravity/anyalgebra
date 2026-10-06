"""Pinned values and negative controls for the Albert and BCH layers.

Each test here fails for a specific wrong implementation that the earlier
suite accepted: a rescaled cubic norm, a trace form without the product, a
predicate that always answers yes, a BCH series with a wrong cubic term.
"""

from __future__ import annotations

from fractions import Fraction
import random

import pytest

from anyalgebra.experimental.bch import (
    bch_hall_terms,
    bch_homogeneous_terms,
    sum_bch_homogeneous_terms,
)
from anyalgebra.experimental.exceptional import (
    albert_basis,
    albert_derivation_dimension_certificate,
    albert_determinant,
    albert_freudenthal_product,
    albert_identity,
    albert_jordan_product,
    albert_trace_form,
    apply_operator,
    f4_inner_derivation_basis,
    is_albert_derivation,
    is_trace_form_skew,
    left_multiplication_operator,
    one_parameter_action,
    operator_coordinates,
)


Vector = tuple[Fraction, ...]


def _diagonal(first: int, second: int, third: int) -> Vector:
    values = [Fraction() for _ in range(27)]
    values[0], values[17], values[26] = (
        Fraction(first),
        Fraction(second),
        Fraction(third),
    )
    return tuple(values)


def _random_albert(rng: random.Random) -> Vector:
    return tuple(Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(27))


def test_cubic_norm_and_adjoint_have_their_defining_values() -> None:
    identity = tuple(Fraction(entry) for entry in albert_identity())
    assert albert_determinant(identity) == 1
    assert albert_determinant(_diagonal(2, 3, 5)) == 30
    assert albert_determinant(_diagonal(2, 3, 0)) == 0

    rng = random.Random(27)
    for _ in range(4):
        element = _random_albert(rng)
        adjoint = albert_freudenthal_product(element, element)
        norm = albert_determinant(element)
        assert albert_jordan_product(element, adjoint) == tuple(
            norm * entry for entry in identity
        )
        # The cubic norm is homogeneous of degree three.
        assert albert_determinant(tuple(2 * entry for entry in element)) == 8 * norm


def test_trace_form_gram_is_diagonal_with_ones_and_twos() -> None:
    basis = albert_basis()
    diagonal_slots = {0, 17, 26}
    for row, left in enumerate(basis):
        for column, right in enumerate(basis):
            expected = 0 if row != column else (1 if row in diagonal_slots else 2)
            assert albert_trace_form(left, right) == expected


def test_jordan_identity_holds_on_random_rational_elements() -> None:
    rng = random.Random(1933)
    for _ in range(6):
        x = _random_albert(rng)
        y = _random_albert(rng)
        square = albert_jordan_product(x, x)
        assert albert_jordan_product(x, y) == albert_jordan_product(y, x)
        assert albert_jordan_product(albert_jordan_product(x, y), square) == (
            albert_jordan_product(x, albert_jordan_product(y, square))
        )


def test_derivation_predicates_reject_operators_that_are_not_derivations() -> None:
    basis = albert_basis()
    left_multiplication = left_multiplication_operator(basis[1])
    assert not is_albert_derivation(left_multiplication)
    assert not is_trace_form_skew(left_multiplication)
    with pytest.raises(ValueError, match="outside the declared span"):
        operator_coordinates(left_multiplication, f4_inner_derivation_basis())

    derivation = f4_inner_derivation_basis()[0]
    assert is_albert_derivation(derivation)
    assert is_trace_form_skew(derivation)
    coordinates = operator_coordinates(derivation, f4_inner_derivation_basis())
    assert coordinates == tuple(Fraction(int(index == 0)) for index in range(52))
    with pytest.raises(TypeError, match="Fraction"):
        is_albert_derivation(
            tuple(tuple(float(entry) for entry in row) for row in derivation)
        )


def test_dimension_certificate_requires_a_real_prime() -> None:
    certificate = albert_derivation_dimension_certificate()
    assert (certificate.rational_rank, certificate.nullity) == (677, 52)
    assert certificate.modular_rank == certificate.rational_rank
    for composite in (4, 9, 15, 1_000_001):
        with pytest.raises(ValueError, match="prime"):
            albert_derivation_dimension_certificate(composite)
    with pytest.raises(ValueError, match="prime"):
        albert_derivation_dimension_certificate(2)
    assert albert_derivation_dimension_certificate(65_537).modular_rank == 677


def test_one_parameter_action_is_a_nontrivial_accurate_exponential() -> None:
    generator = f4_inner_derivation_basis()[0]
    element = tuple(Fraction((index % 5) - 2, (index % 3) + 1) for index in range(27))
    parameter = Fraction(37, 100)

    exact = list(element)
    term: tuple[Fraction, ...] = element
    for order in range(1, 26):
        applied = apply_operator(generator, term)
        term = tuple(parameter * Fraction(entry) / order for entry in applied)
        exact = [total + entry for total, entry in zip(exact, term, strict=True)]

    numerical = one_parameter_action(generator, element, float(parameter))
    assert (
        max(
            abs(got - float(expected))
            for got, expected in zip(numerical, exact, strict=True)
        )
        < 1e-13
    )
    assert (
        max(
            abs(got - float(start))
            for got, start in zip(numerical, element, strict=True)
        )
        > 1e-2
    )

    # Large parameters and long series no longer overflow or lose accuracy.
    far = one_parameter_action(generator, element, 40.0, terms=200)
    norm = float(albert_trace_form(element, element))
    far_exact = tuple(Fraction(entry) for entry in far)
    assert abs(float(albert_trace_form(far_exact, far_exact)) - norm) < 1e-9 * norm
    with pytest.raises(ValueError, match="finite"):
        one_parameter_action(generator, element, float("nan"))
    with pytest.raises(ArithmeticError, match="did not converge"):
        one_parameter_action(generator, element, 0.9, terms=2)


def _matrix_bracket(size: int):  # type: ignore[no-untyped-def]
    def bracket(left, right):  # type: ignore[no-untyped-def]
        result = []
        for row in range(size):
            for column in range(size):
                total = Fraction()
                for inner in range(size):
                    total += left[row * size + inner] * right[inner * size + column]
                    total -= right[row * size + inner] * left[inner * size + column]
                result.append(total)
        return tuple(result)

    return bracket


def _matrix_product(left: Vector, right: Vector, size: int) -> Vector:
    return tuple(
        sum(
            (
                left[row * size + inner] * right[inner * size + column]
                for inner in range(size)
            ),
            Fraction(),
        )
        for row in range(size)
        for column in range(size)
    )


def test_bch_through_degree_four_is_exact_on_nilpotent_matrices() -> None:
    size = 5
    identity = tuple(
        Fraction(int(row == column)) for row in range(size) for column in range(size)
    )

    def exponential(value: Vector) -> Vector:
        result, power = identity, identity
        for order in range(1, size):
            power = tuple(
                entry / order for entry in _matrix_product(power, value, size)
            )
            result = tuple(a + b for a, b in zip(result, power, strict=True))
        return result

    def logarithm(value: Vector) -> Vector:
        shifted = tuple(a - b for a, b in zip(value, identity, strict=True))
        result = tuple(Fraction() for _ in identity)
        power = identity
        for order in range(1, size):
            power = _matrix_product(power, shifted, size)
            sign = 1 if order % 2 else -1
            result = tuple(
                total + Fraction(sign, order) * entry
                for total, entry in zip(result, power, strict=True)
            )
        return result

    rng = random.Random(4)
    quartic_terms_seen = 0
    bracket = _matrix_bracket(size)
    for _ in range(8):
        # Strictly upper triangular 5-by-5 matrices: every product of five
        # vanishes, so the series stops exactly at degree four.
        x = tuple(
            Fraction(rng.randint(-3, 3), rng.randint(1, 3))
            if column > row
            else Fraction()
            for row in range(size)
            for column in range(size)
        )
        y = tuple(
            Fraction(rng.randint(-3, 3), rng.randint(1, 3))
            if column > row
            else Fraction()
            for row in range(size)
            for column in range(size)
        )
        expected = logarithm(_matrix_product(exponential(x), exponential(y), size))
        terms = bch_homogeneous_terms(x, y, bracket, degree=4)
        assert sum_bch_homogeneous_terms(terms) == expected
        # A series that drops its quartic term must not pass this test.
        if any(terms[3]):
            quartic_terms_seen += 1
            assert sum_bch_homogeneous_terms(terms[:3]) != expected
    assert quartic_terms_seen


def test_bch_evaluator_uses_the_certified_hall_coefficients_and_checks_shapes() -> None:
    coefficients = {term.expression: term.coefficient for term in bch_hall_terms(4)}
    assert coefficients == {
        "X": 1,
        "Y": 1,
        "[X,Y]": Fraction(1, 2),
        "[X,[X,Y]]": Fraction(1, 12),
        "[Y,[Y,X]]": Fraction(1, 12),
        "[Y,[X,[X,Y]]]": Fraction(-1, 24),
    }
    with pytest.raises(ValueError, match="same dimension"):
        bch_homogeneous_terms(
            (Fraction(1),), (Fraction(1), Fraction(2)), lambda a, b: ()
        )
    with pytest.raises(ValueError, match="changed the vector dimension"):
        bch_homogeneous_terms(
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            lambda a, b: (Fraction(1),),
        )
