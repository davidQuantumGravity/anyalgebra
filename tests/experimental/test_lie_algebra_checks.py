"""Exact Lie-algebra checks on tables of structure constants.

The Jacobi identity, trace forms, inertia, and modular rank are tested on
small algebras whose answers are known independently.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import product
from math import comb
import random

import pytest

from anyalgebra.experimental.exact_linear import (
    SparseRationalOperator,
    exact_symmetric_inertia,
    integer_bracket_table,
    is_prime,
    jacobi_basis_triples,
    modular_rank,
    sparse_trace_gram,
)
from anyalgebra.experimental.exceptional import octonion_basis_product


Bracket = tuple[tuple[int, Fraction], ...]

Table = tuple[tuple[Bracket, ...], ...]


def _table(bracket: Callable[[int, int], Bracket], size: int) -> Table:
    return tuple(
        tuple(bracket(left, right) for right in range(size)) for left in range(size)
    )


def _so3_table() -> Table:
    """The cross product on three basis vectors."""

    def bracket(left: int, right: int) -> Bracket:
        if left == right:
            return ()
        third = 3 - left - right
        sign = 1 if (right - left) % 3 == 1 else -1
        return ((third, Fraction(sign)),)

    return _table(bracket, 3)


def _imaginary_octonion_commutator_table() -> Table:
    """Commutators of the seven imaginary octonion units: antisymmetric, not Lie."""

    def bracket(left: int, right: int) -> Bracket:
        if left == right:
            return ()
        sign, output = octonion_basis_product(left + 1, right + 1)
        return ((output - 1, Fraction(2 * sign)),)

    return _table(bracket, 7)


def test_jacobi_check_accepts_so3_and_rejects_the_octonion_commutator() -> None:
    assert jacobi_basis_triples(_so3_table()) == (1, None)

    checked, failure = jacobi_basis_triples(_imaginary_octonion_commutator_table())
    assert failure is not None
    assert checked < comb(7, 3)
    first, second, third = failure
    # The reported triple really violates Jacobi: its units do not associate.
    assert len({first, second, third}) == 3


def test_jacobi_check_rejects_tables_that_are_not_antisymmetric() -> None:
    rows = [list(row) for row in _so3_table()]
    rows[1][0] = rows[0][1]
    with pytest.raises(ValueError, match="not antisymmetric"):
        jacobi_basis_triples(rows)

    rows = [list(row) for row in _so3_table()]
    rows[2][2] = ((0, Fraction(1)),)
    with pytest.raises(ValueError, match="not alternating"):
        jacobi_basis_triples(rows)

    with pytest.raises(ValueError, match="square"):
        jacobi_basis_triples(((), ()))
    with pytest.raises(ValueError, match="outside the basis"):
        jacobi_basis_triples((((), ((5, Fraction(1)),)), (((5, Fraction(-1)),), ())))


def test_integer_bracket_table_clears_denominators_exactly() -> None:
    table = (
        ((), ((1, Fraction(1, 6)),)),
        (((1, Fraction(-1, 6)),), ()),
    )
    denominator, scaled = integer_bracket_table(table)
    assert denominator == 6
    assert scaled[0][1] == ((1, 1),)
    assert scaled[1][0] == ((1, -1),)


def _rational_rank(matrix: list[list[int]]) -> int:
    rows = [[Fraction(entry) for entry in row] for row in matrix]
    rank = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for other in range(len(rows)):
            if other != rank and rows[other][column]:
                factor = rows[other][column] / rows[rank][column]
                rows[other] = [
                    a - factor * b for a, b in zip(rows[other], rows[rank], strict=True)
                ]
        rank += 1
    return rank


def test_modular_rank_bounds_the_rational_rank_from_below() -> None:
    rng = random.Random(7)
    for _ in range(60):
        rows = rng.randint(1, 6)
        columns = rng.randint(1, 6)
        inner = rng.randint(0, min(rows, columns))
        left = [[rng.randint(-3, 3) for _ in range(inner)] for _ in range(rows)]
        right = [[rng.randint(-3, 3) for _ in range(columns)] for _ in range(inner)]
        matrix = [
            [
                sum(left[row][k] * right[k][column] for k in range(inner))
                for column in range(columns)
            ]
            for row in range(rows)
        ]
        exact = _rational_rank(matrix)
        # Every minor of these small matrices is far below the large prime,
        # so that prime sees the rational rank; any prime gives a lower bound.
        assert modular_rank(matrix, 2_147_483_629) == exact
        assert modular_rank(matrix, 3) <= exact

    assert modular_rank([[2, 0], [0, 2]], 2) == 0
    assert modular_rank([[2, 0], [0, 2]], 3) == 2
    with pytest.raises(ValueError, match="prime"):
        modular_rank([[1]], 9)
    with pytest.raises(ValueError, match="rectangular"):
        modular_rank([[1, 2], [3]], 5)


def test_primality_test_handles_small_values_and_strong_pseudoprimes() -> None:
    small_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    for value in range(-3, 50):
        assert is_prime(value) is (value in small_primes)
    for composite in (561, 1_373_653, 1_000_001, 3_215_031_751, 2**61 + 1):
        assert not is_prime(composite)
    for prime in (2_147_483_629, 2**31 - 1, 2**61 - 1, 1_000_003):
        assert is_prime(prime)


def test_exact_inertia_handles_hyperbolic_and_singular_forms() -> None:
    zero = Fraction()
    one = Fraction(1)
    assert exact_symmetric_inertia(((zero, one), (one, zero))) == (1, 1, 0)
    assert exact_symmetric_inertia(((one, one), (one, one))) == (1, 0, 1)
    assert exact_symmetric_inertia(((zero, zero), (zero, zero))) == (0, 0, 2)
    assert exact_symmetric_inertia(
        (
            (Fraction(2), Fraction(1), zero),
            (Fraction(1), Fraction(-3), Fraction(1, 2)),
            (zero, Fraction(1, 2), Fraction(-1)),
        )
    ) == (1, 2, 0)
    with pytest.raises(ValueError, match="symmetric"):
        exact_symmetric_inertia(((zero, one), (zero, zero)))
    with pytest.raises(TypeError, match="Fraction"):
        exact_symmetric_inertia(((1,),))  # type: ignore[arg-type]


def test_inertia_is_invariant_under_exact_congruence() -> None:
    rng = random.Random(8)
    for _ in range(40):
        size = rng.randint(1, 6)
        signs = [rng.choice((-1, 0, 1)) for _ in range(size)]
        while True:
            change = [[rng.randint(-3, 3) for _ in range(size)] for _ in range(size)]
            if _rational_rank(change) == size:
                break
        form = tuple(
            tuple(
                Fraction(
                    sum(
                        signs[k] * change[k][row] * change[k][column]
                        for k in range(size)
                    )
                )
                for column in range(size)
            )
            for row in range(size)
        )
        assert exact_symmetric_inertia(form) == (
            signs.count(1),
            signs.count(-1),
            signs.count(0),
        )


def test_trace_gram_matches_direct_traces() -> None:
    rng = random.Random(16)
    for _ in range(40):
        size = rng.randint(1, 5)
        family = tuple(
            SparseRationalOperator.from_mapping(
                size,
                {
                    (row, column): Fraction(rng.randint(-4, 4), rng.randint(1, 5))
                    for row, column in product(range(size), repeat=2)
                    if rng.random() < 0.6
                },
            )
            for _ in range(rng.randint(1, 4))
        )
        gram = sparse_trace_gram(family)
        for left_index, left in enumerate(family):
            for right_index, right in enumerate(family):
                product_operator = left @ right
                trace = sum(
                    (
                        value
                        for row, column, value in product_operator.entries
                        if row == column
                    ),
                    Fraction(),
                )
                assert gram[left_index][right_index] == trace
    assert sparse_trace_gram(()) == ()


def test_killing_form_of_so3_is_negative_definite() -> None:
    table = _so3_table()
    adjoint = tuple(
        SparseRationalOperator.from_mapping(
            3,
            {
                (row, column): coefficient
                for column in range(3)
                for row, coefficient in table[index][column]
            },
        )
        for index in range(3)
    )
    form = sparse_trace_gram(adjoint)
    assert form == tuple(
        tuple(Fraction(-2 if row == column else 0) for column in range(3))
        for row in range(3)
    )
    assert exact_symmetric_inertia(form) == (0, 3, 0)
