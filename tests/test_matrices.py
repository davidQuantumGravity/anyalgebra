"""Matrices over algebras: ordered products, conjugations, and bases.

Dimension counts are the classical ones: the Hermitian n-by-n matrices over a
composition algebra of dimension d have real dimension n + d n(n-1)/2, and the
anti-Hermitian ones n(d-1) + d n(n-1)/2.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.composition as ca
import anyalgebra.easy as aa
import anyalgebra.matrices as am


def test_arithmetic_and_ordered_products_over_the_quaternions() -> None:
    quaternions = ca.quaternions()
    i, j, k = quaternions.e1, quaternions.e2, quaternions.e3
    x = am.matrix(quaternions, [[1, i], [j, 0]])
    y = am.matrix(quaternions, [[k, 0], [0, 2]])
    assert (x @ y) == am.matrix(quaternions, [[k, 2 * i], [j * k, 0]])
    assert (y @ x) == am.matrix(quaternions, [[k, k * i], [2 * j, 0]])
    assert am.comm(x, y) == x @ y - y @ x and am.anticomm(x, y) == x @ y + y @ x
    assert am.jordan(x, y) == am.jordan(y, x) == (x @ y + y @ x) / 2
    assert x + y - y == x and -x + x == am.zeros(quaternions, 2) and not (x - x)
    assert 2 * x == x * 2 == x + x and x / 2 + x / 2 == x
    assert i * x == am.matrix(quaternions, [[i, -1], [k, 0]])
    assert x * i == am.matrix(quaternions, [[i, -1], [-k, 0]])
    assert x.shape == (2, 2) and x[0, 1] == i and [row[0] for row in x] == [1, j]
    assert x.transpose() == am.matrix(quaternions, [[1, j], [i, 0]])
    assert x.dagger() == am.matrix(quaternions, [[1, -j], [-i, 0]])
    assert x.trace() == 1 and am.identity(quaternions, 2) @ x == x
    assert x != y and x != "x" and bool(x)
    assert am.coordinates(x) == tuple(
        Fraction(v) for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    )


def test_hermitian_and_antihermitian_predicates_and_inner_product() -> None:
    quaternions = ca.quaternions()
    i = quaternions.e1
    hermitian = am.matrix(quaternions, [[1, i], [-i, 2]])
    skew = am.matrix(quaternions, [[i, 1], [-1, 0]])
    assert hermitian.is_hermitian() and not hermitian.is_antihermitian()
    assert skew.is_antihermitian() and not skew.is_hermitian()
    assert not am.zeros(quaternions, 1, 2).is_hermitian()
    assert am.inner(hermitian, hermitian) == 1 + 1 + 1 + 4
    assert am.inner(hermitian, skew) == am.inner(skew, hermitian)
    plain = (1,) * 4
    assert hermitian.conj(plain) == hermitian and hermitian.dagger(plain) != hermitian
    assert not hermitian.is_hermitian(plain) and not skew.is_antihermitian(plain)


@pytest.mark.parametrize(
    ("make", "dimension"),
    [(ca.reals, 1), (ca.complexes, 2), (ca.quaternions, 4), (ca.octonions, 8)],
)
def test_basis_sizes_match_the_classical_dimension_counts(
    make: Callable[[], aa.Algebra], dimension: int
) -> None:
    source = make()
    for size in (1, 2, 3):
        pairs = size * (size - 1) // 2
        hermitian = am.hermitian_basis(source, size)
        skew = am.antihermitian_basis(source, size)
        assert len(hermitian) == size + dimension * pairs
        assert len(skew) == size * (dimension - 1) + dimension * pairs
        assert all(x.is_hermitian() for x in hermitian)
        assert all(x.is_antihermitian() for x in skew)
        assert len(hermitian) + len(skew) == dimension * size * size


def test_octonionic_matrix_products_are_not_associative() -> None:
    octonions = ca.octonions()
    e = octonions.basis
    x = am.unit(octonions, 2, 0, 1, e[1])
    y = am.unit(octonions, 2, 1, 0, e[2])
    z = am.unit(octonions, 2, 0, 0, e[4])
    assert (x @ y) @ z != x @ (y @ z)
    assert ((x @ y) @ z - x @ (y @ z))[0, 0] == aa.assoc(e[1], e[2], e[4])


def test_print_forms() -> None:
    quaternions = ca.quaternions()
    x = am.matrix(quaternions, [[1, quaternions.e1], [-quaternions.e1, 2]])
    assert str(x) == "[  1  e1]\n[-e1   2]"
    assert f"{x:v}" == "[ [1, 0, 0, 0]  [0, 1, 0, 0]]\n[[0, -1, 0, 0]  [2, 0, 0, 0]]"
    assert f"{x:l}" == r"\begin{pmatrix} 1 & e_{1} \\ -e_{1} & 2 \end{pmatrix}"
    assert x._repr_latex_() == "$" + f"{x:l}" + "$"
    assert repr(x) == "Mat(H, 2x2)"
    with aa.display("latex"):
        assert str(x) == f"{x:l}"
    with aa.display("vector"):
        assert str(x) == f"{x:v}"


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    quaternions, other = ca.quaternions(), ca.quaternions()
    x = am.identity(quaternions, 2)
    no_unit = aa.algebra(("a",), {("a", "a"): "a"}, unit=None)
    plain = aa.algebra(("1", "a"), {("1", "1"): "1"})
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: am.matrix(quaternions, []), "shape"),
        (lambda: am.matrix(quaternions, [[1, 2], [3]]), "shape"),
        (lambda: am.matrix(quaternions, [[other.e1]]), "parent"),
        (lambda: am.matrix(quaternions, [["x"]]), "parent"),  # type: ignore[list-item]
        (lambda: x + am.identity(other, 2), "parent"),
        (lambda: x + "y", "parent"),
        (lambda: x + am.identity(quaternions, 3), "shape"),
        (lambda: x @ am.identity(quaternions, 3), "shape"),
        (lambda: am.zeros(quaternions, 1, 2).trace(), "shape"),
        (lambda: am.unit(quaternions, 2, 2, 0), "shape"),
        (lambda: am.hermitian_basis(plain, 2), "involution"),
        (lambda: am.inner(am.unit(no_unit, 1, 0, 0, no_unit.a), x), "parent"),
        (lambda: am.identity(no_unit, 1), "unit"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
    with pytest.raises(TypeError):
        x * "y"
    with pytest.raises(TypeError):
        "y" * x
    signed = aa.wrap(no_unit.structure, unit=None, conjugation_signs=(1,))
    lonely = am.unit(signed, 1, 0, 0, signed.a)
    with pytest.raises(aa.EasyError) as caught:
        am.inner(lonely, lonely)
    assert caught.value.code == "unit"
