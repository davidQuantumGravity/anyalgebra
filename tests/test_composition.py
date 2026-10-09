"""Cayley--Dickson algebras, involutions and tensor products against known facts.

The expected properties are textbook results (Schafer, *An Introduction to
Nonassociative Algebras*; Baez, *The Octonions*): the doubling chain loses
commutativity, then associativity, then alternativity and the composition
law, in that order, and Der(H) and Der(O) have dimensions 3 and 14.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import product

import pytest

import anyalgebra.composition as ca
import anyalgebra.easy as aa

Maker = Callable[[], aa.Algebra]

# name, constructor, rank, commutative, associative, alternative, composition
CHAIN: tuple[tuple[str, Maker, int, bool, bool, bool, bool], ...] = (
    ("R", ca.reals, 1, True, True, True, True),
    ("C", ca.complexes, 2, True, True, True, True),
    ("Cs", ca.split_complexes, 2, True, True, True, True),
    ("H", ca.quaternions, 4, False, True, True, True),
    ("Hs", ca.split_quaternions, 4, False, True, True, True),
    ("O", ca.octonions, 8, False, False, True, True),
    ("Os", ca.split_octonions, 8, False, False, True, True),
    ("S", ca.sedenions, 16, False, False, False, False),
)


@pytest.mark.parametrize("row", CHAIN, ids=[row[0] for row in CHAIN])
def test_the_doubling_chain_has_the_textbook_properties(
    row: tuple[str, Maker, int, bool, bool, bool, bool],
) -> None:
    name, make, rank, commutative, associative, alternative, composition = row
    found = make()
    assert (found.name, found.rank) == (name, rank)
    assert found.labels == ("1", *(f"e{n}" for n in range(1, rank)))
    assert ca.is_commutative(found) is commutative
    assert ca.is_associative(found) is associative
    assert ca.is_alternative(found) is alternative
    assert ca.is_composition(found) is composition
    assert ca.is_involution(found)
    for element in found.basis:
        assert element * found.one == found.one * element == element


def test_norms_are_definite_on_division_forms_and_isotropic_on_split_forms() -> None:
    for make in (ca.complexes, ca.quaternions, ca.octonions):
        found = make()
        assert all(element.norm() == 1 for element in found.basis)
    for make in (ca.split_complexes, ca.split_quaternions, ca.split_octonions):
        found = make()
        norms = sorted(element.norm() for element in found.basis)
        half = found.rank // 2
        assert norms == [-1] * half + [1] * half
        isotropic = found.basis[0] + found.basis[half]
        assert isotropic.norm() == 0 and isotropic * isotropic.conj() == 0


def test_derivation_dimensions_match_the_pinned_fixtures() -> None:
    assert "derivation dimension  3" in ca.quaternions().report()
    assert "derivation dimension  14" in ca.octonions().report()
    assert "derivation dimension  14" in ca.split_octonions().report()


def test_the_sedenions_have_zero_divisors() -> None:
    sedenions = ca.sedenions()
    basis = sedenions.basis
    found = next(
        (
            (basis[1] + b, c + sign * d)
            for b, c, d in product(basis[2:], repeat=3)
            for sign in (1, -1)
            if len({b.vector, c.vector, d.vector}) == 3
            and (basis[1] + b) * (c + sign * d) == 0
        ),
        None,
    )
    assert found is not None, "no zero divisor (e1 + eb)(ec +- ed) was found"
    x, y = found
    assert x != 0 and y != 0 and x * y == 0 and x.norm() == 2


def test_inverses_and_division_in_the_octonions() -> None:
    octonions = ca.octonions()
    e = octonions.basis
    a = 1 + 2 * e[1] - e[4] + Fraction(1, 2) * e[7]
    b = 3 * e[2] + e[5]
    assert a * a.inv() == a.inv() * a == 1
    assert a * ca.left_divide(a, b) == b
    assert ca.right_divide(b, a) * a == b
    assert ca.inner(a, a) == a.norm() and ca.inner(e[1], e[2]) == 0


def test_tensor_products_and_factor_conjugations() -> None:
    complexes, quaternions, octonions = ca.complexes(), ca.quaternions(), ca.octonions()
    ch = ca.tensor(complexes, quaternions)
    assert (ch.name, ch.rank) == ("CxH", 8)
    assert ch.labels == ("1", "f1", "f2", "f3", "e1", "e1.f1", "e1.f2", "e1.f3")
    assert ca.is_associative(ch) and not ca.is_commutative(ch)
    hh = ca.tensor(quaternions, quaternions, name="HH")
    assert hh.rank == 16 and ca.is_associative(hh)
    co = ca.tensor(complexes, octonions)
    assert ca.is_alternative(co) and not ca.is_associative(co)
    for first, second in product((False, True), repeat=2):
        signs = ca.factor_conjugation(
            complexes, quaternions, first=first, second=second
        )
        # Conjugating the commutative factor alone is an automorphism, and an
        # anti-automorphism only together with the quaternionic conjugation.
        assert ca.is_involution(ch, signs) is second
    assert ca.is_involution(ch)
    x = ch.f1 + 2 * ch[5]
    only_first = ca.factor_conjugation(complexes, quaternions, first=True)
    assert ca.conjugate(x, only_first) == ch.f1 - 2 * ch[5]
    # C tensor C has zero divisors although both factors are fields.
    cc = ca.tensor(complexes, complexes)
    i1, i2 = cc.e1, cc.f1
    assert (1 + i1 * i2) * (1 - i1 * i2) == 0
    assert ca.is_composition(cc) is False


def test_automorphisms_of_the_quaternions() -> None:
    quaternions = ca.quaternions()
    one, i, j, k = quaternions.basis
    assert i * j == k
    assert ca.is_automorphism(quaternions, (one, j, k, i))
    assert ca.is_automorphism(quaternions, (one, -i, -j, k))
    assert not ca.is_automorphism(quaternions, (one, j, i, k))
    assert not ca.is_automorphism(quaternions, (one, i, i, k))
    assert not ca.is_automorphism(quaternions, (one, quaternions.zero, j, k))
    # A multiplicative map that is not invertible is not an automorphism.
    reals = ca.reals()
    assert not ca.is_automorphism(reals, (reals.zero,))
    assert ca.is_automorphism(reals, (reals.one,))


def _signed(labels: tuple[str, ...]) -> aa.Algebra:
    plain = aa.algebra(labels, {("1", "1"): "1"})
    signs = (1, *(-1 for _ in labels[1:]))
    return aa.wrap(plain.structure, conjugation_signs=signs)


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    plain = aa.algebra(("1", "a"), {("1", "1"): "1", ("1", "a"): "a", ("a", "1"): "a"})
    no_unit = aa.algebra(("a",), {("a", "a"): "a"}, unit=None)
    quaternions = ca.quaternions()
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: ca.cayley_dickson(plain, name="X"), "involution"),
        (lambda: ca.cayley_dickson(quaternions, gamma=0, name="X"), "involution"),
        (lambda: ca.tensor(no_unit, quaternions), "unit"),
        (lambda: ca.factor_conjugation(plain, quaternions, first=True), "involution"),
        (lambda: ca.conjugate(quaternions.one, (1, 1)), "involution"),
        (lambda: ca.is_involution(plain), "involution"),
        (lambda: ca.is_automorphism(quaternions, (quaternions.one,)), "coordinates"),
        (
            lambda: ca.tensor(
                _signed(("1", "a", "a.b")),
                _signed(("1", "b")),
            ),
            "table",
        ),
        (lambda: ca.inner(no_unit.basis[0], no_unit.basis[0]), "involution"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
    assert ca.is_composition(plain) is False
    odd = aa.wrap(
        aa.algebra(("1", "a"), {("1", "1"): "1", ("a", "a"): "a"}).structure,
        conjugation_signs=(1, 1),
    )
    with pytest.raises(aa.EasyError) as caught:
        ca.inner(odd.a, odd.a)
    assert caught.value.code == "norm"
    assert ca.is_composition(odd) is False
    shared = ca.tensor(_signed(("1", "x")), _signed(("1", "x")))
    assert shared.labels == ("1", "x_2", "x", "x.x_2")
