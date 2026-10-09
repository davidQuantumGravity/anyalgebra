"""Clifford algebras and gamma matrices against the classical facts.

Lawson and Michelsohn, *Spin Geometry*, ch. I: Cl(p,q) has dimension 2^n; its
center is spanned by 1, and also by the pseudoscalar when n is odd; the
pseudoscalar squares to (-1)^(n(n-1)/2 + q); Cl(0,1) = C, Cl(0,2) = H,
Cl(2,0) = M_2(R); and the even subalgebra of Cl(3,0) is H.  In odd dimension
2m+1 the charge conjugation obeys C g C^-1 = (-1)^m g^T.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import combinations

import pytest

import anyalgebra.clifford as ac
import anyalgebra.composition as ca
import anyalgebra.easy as aa
import anyalgebra.lie as al

SIGNATURES = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (1, 3), (0, 3), (4, 1))


@pytest.mark.parametrize(("p", "q"), SIGNATURES)
def test_dimension_squares_center_and_pseudoscalar(p: int, q: int) -> None:
    n = p + q
    found = ac.clifford(p, q)
    assert found.rank == 2**n and found.labels[: n + 1] == (
        "1",
        *(f"e{k}" for k in range(1, n + 1)),
    )
    generators = found.basis[1 : n + 1]
    assert [g * g for g in generators] == [1] * p + [-1] * q
    assert all(a * b == -(b * a) for a, b in combinations(generators, 2))
    assert n > 3 or ca.is_associative(found)
    omega = ac.pseudoscalar(found)
    assert omega * omega == (-1) ** (n * (n - 1) // 2 + q)
    central = [x for x in found.basis if all(x * g == g * x for g in generators)]
    assert central == ([found.one, omega] if n % 2 else [found.one])


def test_small_algebras_are_the_classical_ones() -> None:
    complex_like = ac.clifford(0, 1)
    assert ca.is_commutative(complex_like) and complex_like.e1**2 == -1
    quaternion_like = ac.clifford(0, 2)
    signed = aa.wrap(
        quaternion_like.structure,
        conjugation_signs=ac.clifford_conjugation(quaternion_like),
    )
    assert ca.is_composition(signed) and not ca.is_commutative(signed)
    assert all(x.norm() == 1 for x in signed.basis)
    matrices = ac.clifford(2, 0)
    assert (1 + matrices.e1) * (1 - matrices.e1) == 0
    even = ac.even_subalgebra(ac.clifford(3, 0))
    assert even.labels == ("1", "e12", "e13", "e23") and even.name == "Cl(3,0)+"
    assert ca.is_composition(even) and ca.is_associative(even)
    assert all(x.norm() == 1 for x in even.basis)


def test_involutions_and_grades() -> None:
    found = ac.clifford(1, 3)
    assert ac.grades(found)[:6] == (0, 1, 1, 1, 1, 2)
    assert sorted(set(ac.grades(found))) == [0, 1, 2, 3, 4]
    assert found.conjugation_signs == ac.reversion(found)
    assert ca.is_involution(found) and ca.is_involution(
        found, ac.clifford_conjugation(found)
    )
    assert not ca.is_involution(found, ac.grade_involution(found))
    x = 2 + found.e1 - 3 * found.e12 + found.e123
    assert ac.grade_part(x, 2) == -3 * found.e12 and ac.grade_part(x, 4) == 0
    assert x.conj() == 2 + found.e1 + 3 * found.e12 - found.e123
    assert (
        ca.conjugate(x, ac.grade_involution(found))
        == 2 - found.e1 - 3 * found.e12 - found.e123
    )
    assert found.e1 * found.e2 == found.e12 and found.e2 * found.e1 == -found.e12


def test_grassmann_algebra() -> None:
    found = ac.grassmann(3)
    assert found.name == "Gr(3)" and found.rank == 8
    assert found.e1 * found.e1 == 0 and found.e1 * found.e2 == -(found.e2 * found.e1)
    assert found.e1 * found.e2 * found.e3 == found.e123
    assert found.e12 * found.e1 == 0 and ca.is_associative(found)
    assert ac.clifford(1, 0, 1).name == "Cl(1,0,1)"


def test_blade_product_matches_the_table() -> None:
    signature = (1, -1, -1, -1)
    found = ac.clifford(1, 3)
    for a, b in (
        (0b0011, 0b0110),
        (0b1000, 0b0001),
        (0b1111, 0b1111),
        (0b0101, 0b0101),
    ):
        sign, mask = ac.blade_product(a, b, signature)
        left = found[ac._label(a)] * found[ac._label(b)]
        assert left == sign * found[ac._label(mask)]


GAMMA_SIGNATURES = (
    (1, 0),
    (0, 1),
    (2, 0),
    (3, 0),
    (1, 3),
    (3, 1),
    (4, 1),
    (9, 1),
    (11, 3),
    (12, 4),
)


@pytest.mark.parametrize(("p", "q"), GAMMA_SIGNATURES)
def test_gamma_matrices_satisfy_the_clifford_relations(p: int, q: int) -> None:
    n = p + q
    gammas = ac.gamma_matrices(p, q)
    assert len(gammas) == n and all(g.size == 2 ** max(n // 2, 1) for g in gammas)
    assert ac.satisfies_clifford_relations(gammas, p, q)
    assert not ac.satisfies_clifford_relations(gammas, q, p) or p == q
    assert not ac.satisfies_clifford_relations(gammas[:-1], p, q)
    top = ac.chirality(gammas)
    assert (top @ top).is_identity_multiple() == (0, Fraction(1))
    if n % 2 == 0:
        assert all(top @ g == -(g @ top) for g in gammas)
    elif n > 1:
        assert all(top @ g == g @ top for g in gammas)
        assert top.is_identity_multiple() is not None
    found = ac.charge_conjugations(gammas)
    if n % 2 == 0:
        assert set(found) == {1, -1}
    elif n > 1:
        assert set(found) == {(-1) ** (n // 2)}
    for sign, matrix in found.items():
        inverse = matrix.dagger()
        assert all(matrix @ g @ inverse == sign * g.transpose() for g in gammas)


def test_monomial_matrices() -> None:
    s1, s2, s3 = ac.pauli()
    assert s1 @ s2 == s3.times_i() and s2 @ s1 == -(s3.times_i())
    assert [[tuple(map(int, z)) for z in row] for row in s2.dense()] == [
        [(0, 0), (0, -1)],
        [(0, 1), (0, 0)],
    ]
    assert s2.transpose() == -s2 and s2.dagger() == s2 and s1.transpose() == s1
    assert (2 * s3).scale == 2 and (s3 * Fraction(1, 2)) @ (2 * s3) == ac.identity(2)
    assert ac.kron(s1, s3).columns == (2, 3, 0, 1) and ac.kron(s1, s3).phases == (
        0,
        2,
        0,
        2,
    )
    assert [list(map(int, row)) for row in s2.realified()] == [
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
    ]
    assert s1 != s3 and (s1 == "s1") is False and repr(s1) == "Monomial(size=2)"
    assert s1 * 0 == s3 * 0 and s1 * 0 != s1 and s3.is_identity_multiple() is None
    assert s1.is_identity_multiple() is None
    assert (-ac.identity(2)).is_identity_multiple() == (2, Fraction(1))


@pytest.mark.parametrize(("p", "q"), ((3, 0), (2, 1), (3, 1), (4, 0), (4, 1)))
def test_spin_algebra_is_the_orthogonal_algebra(p: int, q: int) -> None:
    spin, orthogonal = ac.spin_algebra(p, q), al.so(p, q)
    assert spin.dimension == orthogonal.dimension
    assert spin.jacobi_holds()
    assert spin.killing_signature() == orthogonal.killing_signature()


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    s1, _, _ = ac.pauli()
    plain = aa.algebra(("1", "a"), {("1", "1"): "1"})
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: ac.clifford(0, 0), "signature"),
        (lambda: ac.clifford(9, 0), "signature"),
        (lambda: ac.clifford(-1, 2), "signature"),
        (lambda: ac.gamma_matrices(0, 0), "signature"),
        (lambda: ac.gamma_matrices(-1, 2), "signature"),
        (lambda: ac.grades(plain), "signature"),
        (lambda: ac.Monomial((0, 0), (0, 0)), "shape"),
        (lambda: ac.Monomial((0, 1), (0,)), "shape"),
        (lambda: s1 @ ac.identity(4), "shape"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
