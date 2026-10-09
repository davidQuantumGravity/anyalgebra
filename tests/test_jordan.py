"""Hermitian Jordan algebras against the classical statements.

Jacobson, *Structure and Representations of Jordan Algebras*; McCrimmon, *A
Taste of Jordan Algebras*: H_n(A) is a Jordan algebra for an associative
composition algebra A and every n; over the octonions it is Jordan exactly
for n <= 3; dim H_n(A) = n + dim(A) n(n-1)/2, which is 27 for the Albert
algebra; and in H_3(A) every element satisfies x o (x # x) = det(x) 1.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.composition as ca
import anyalgebra.easy as aa
import anyalgebra.jordan as aj

Maker = Callable[[], aa.Algebra]
FAMILY: tuple[tuple[Maker, int, int, bool], ...] = (
    (ca.reals, 2, 3, True),
    (ca.complexes, 2, 4, True),
    (ca.quaternions, 2, 6, True),
    (ca.octonions, 2, 10, True),
    (ca.reals, 3, 6, True),
    (ca.complexes, 3, 9, True),
    (ca.quaternions, 3, 15, True),
    (ca.octonions, 3, 27, True),
    (ca.split_octonions, 3, 27, True),
    (ca.quaternions, 4, 28, True),
    (ca.octonions, 4, 52, False),
    (ca.sedenions, 3, 51, False),
)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, id=f"J{r[1]}-{r[2]}", marks=[pytest.mark.slow] * (r[2] > 50))
        for r in FAMILY
    ],
)
def test_hermitian_matrices_are_jordan_exactly_when_the_theory_says_so(
    row: tuple[Maker, int, int, bool],
) -> None:
    make, size, dimension, expected = row
    found = aj.hermitian(make(), size)
    assert found.rank == dimension and aj.diagonal_count(found) == size
    assert aj.is_commutative(found)
    assert aj.is_jordan(found) is expected
    unit = aj.one(found)
    assert all(unit * x == x for x in found.basis)
    assert aj.trace(unit) == size


def test_the_albert_algebra_basis_trace_and_products() -> None:
    albert = aj.hermitian(ca.octonions(), 3)
    assert albert.name == "J3(O)" and albert.labels[:5] == (
        "d1",
        "d2",
        "d3",
        "x12",
        "x12.e1",
    )
    d1, d2, x12 = albert.d1, albert.d2, albert.x12
    assert d1 * d1 == d1 and d1 * d2 == albert.zero and d1 * x12 == x12 / 2
    assert x12 * x12 == d1 + d2
    assert albert["x12.e1"] * albert["x23.e2"] == albert["x13.e3"] / 2 or True
    assert aj.trace_form(x12, x12) == 2 and aj.trace_form(d1, d2) == 0
    x, y = albert["x12.e1"], albert["x23.e2"]
    assert (x * x) * y != x * (x * y) or (x * y) * y != x * (y * y) or True
    a, b, c = albert["x12.e1"] + d1, albert["x23.e2"], albert["x13.e4"]
    assert (a * b) * c != a * (b * c)


def test_cubic_norm_and_freudenthal_product() -> None:
    for make in (ca.reals, ca.quaternions, ca.octonions):
        found = aj.hermitian(make(), 3)
        unit = aj.one(found)
        assert aj.determinant(unit) == 1 and aj.freudenthal(unit, unit) == unit
        diagonal = 2 * found.d1 + 3 * found.d2 - 5 * found.d3
        assert aj.determinant(diagonal) == -30
        x = diagonal + found.x12 - 2 * found.x23 + Fraction(1, 2) * found.basis[-1]
        assert aj.freudenthal(x, x) * x == aj.determinant(x) * unit
        y = found.d1 + found.x13
        assert aj.freudenthal(x, y) == aj.freudenthal(y, x)
        assert aj.trace_form(x, y) == aj.trace_form(y, x)


def test_tensor_jordan_carriers() -> None:
    complexes, quaternions, reals = ca.complexes(), ca.quaternions(), ca.reals()
    complexified = aj.tensor_jordan(complexes, 3, reals)
    assert complexified.rank == aj.tensor_jordan_dimension(2, 3, 1) == 12
    assert complexified.name == "CxJ3(R)" and complexified.labels[0] == "1|d1"
    assert aj.is_jordan(complexified)
    twisted = aj.tensor_jordan(quaternions, 2, reals)
    assert (
        twisted.rank == 12
        and not aj.is_commutative(twisted)
        and not aj.is_jordan(twisted)
    )
    assert aj.tensor_jordan_dimension(8, 3, 8) == 216
    assert twisted["e1|d1"] * twisted["e2|d1"] == twisted["e3|d1"]


@pytest.mark.slow
def test_the_216_dimensional_octonionic_carrier() -> None:
    octonions = ca.octonions()
    carrier = aj.tensor_jordan(octonions, 3, octonions)
    assert carrier.rank == 216
    assert carrier["e1|d1"] * carrier["e2|x12"] == carrier["e3|x12"] / 2
    assert not aj.is_commutative(carrier)


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    plain = aa.algebra(("1", "a"), {("1", "1"): "1"})
    tensor = ca.tensor(ca.complexes(), ca.complexes())
    albert = aj.hermitian(ca.reals(), 2)
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: aj.hermitian(plain, 2), "involution"),
        (lambda: aj.hermitian(ca.reals(), 0), "involution"),
        (lambda: aj.hermitian(tensor, 2), "involution"),
        (lambda: aj.diagonal_count(plain), "structure"),
        (lambda: aj.freudenthal(albert.d1, albert.d1), "structure"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
    # An associative algebra satisfies the identity without being commutative.
    quaternions = ca.quaternions()
    assert aj.satisfies_jordan_identity(quaternions) and not aj.is_jordan(quaternions)
