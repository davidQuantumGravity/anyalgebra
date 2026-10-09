"""Composition algebras, involutions, and their tensor products.

This module builds the classical algebras with involution exactly, by the
Cayley--Dickson doubling, and checks the properties that make them what they
are instead of assuming them::

    import anyalgebra.composition as ca

    O = ca.octonions()
    print(ca.is_composition(O), ca.is_alternative(O))  # True True
    S = ca.sedenions()
    print(ca.is_composition(S), ca.is_alternative(S))  # False False

An involution is recorded as one sign per basis element.  That covers the
standard conjugation of every Cayley--Dickson algebra and the factor-wise
conjugations of their tensor products.  The returned objects are the handles
of :mod:`anyalgebra.easy`, so elements multiply with ``*`` and print in every
form.

Doubling convention, for an algebra ``A`` with conjugation ``a -> a*`` and a
rational parameter ``gamma``::

    (a, b) (c, d) = (a c + gamma d* b,  d a + b c*)
    (a, b)*       = (a*, -b)

``gamma = -1`` gives the division chain R, C, H, O; ``gamma = +1`` gives a
split form.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import product

from anyalgebra.easy import Algebra, EasyError, Element, algebra

Table = dict[tuple[str, str], dict[str, Fraction]]


def _table(source: Algebra) -> dict[tuple[int, int], tuple[Fraction, ...]]:
    return {
        (i, j): (source.basis[i] * source.basis[j]).vector
        for i in range(source.rank)
        for j in range(source.rank)
    }


def _build(
    name: str,
    labels: Sequence[str],
    products: Mapping[tuple[int, int], Sequence[Fraction]],
    signs: Sequence[int],
) -> Algebra:
    table: dict[tuple[str, str], dict[str, Fraction]] = {}
    for (i, j), vector in products.items():
        cell = {labels[k]: value for k, value in enumerate(vector) if value != 0}
        if cell:
            table[(labels[i], labels[j])] = cell
    unit = "1" if "1" in labels else None
    plain = algebra(tuple(labels), table, name=name, unit=unit)
    return Algebra(plain.structure, name=name, unit=unit, conjugation_signs=signs)


def reals() -> Algebra:
    """Return the one-dimensional algebra of rational scalars."""
    return _build("R", ("1",), {(0, 0): (Fraction(1),)}, (1,))


def cayley_dickson(
    source: Algebra, *, gamma: int | Fraction = -1, name: str
) -> Algebra:
    """Double an algebra with involution by the Cayley--Dickson process.

    The new basis is the old one followed by the old one times a new unit
    ``l`` with ``l*l = gamma``.  Labels are ``1, e1, e2, ...`` in that order.
    """
    signs = source.conjugation_signs
    if signs is None or source.unit_index != 0:
        raise EasyError(
            "involution",
            "doubling needs a declared conjugation and the unit as first basis element",
        )
    if gamma == 0:
        raise EasyError("involution", "the doubling parameter must not be zero")
    scale = Fraction(gamma)
    n = source.rank
    old = _table(source)
    products: dict[tuple[int, int], list[Fraction]] = {
        (row, column): [Fraction(0)] * (2 * n)
        for row in range(2 * n)
        for column in range(2 * n)
    }
    for i, j in product(range(n), repeat=2):
        # (e_i, 0)(e_j, 0) = (e_i e_j, 0)
        for k, value in enumerate(old[(i, j)]):
            products[(i, j)][k] += value
        # (e_i, 0)(0, e_j) = (0, e_j e_i)
        for k, value in enumerate(old[(j, i)]):
            products[(i, n + j)][n + k] += value
        # (0, e_i)(e_j, 0) = (0, e_i e_j*)
        for k, value in enumerate(old[(i, j)]):
            products[(n + i, j)][n + k] += signs[j] * value
        # (0, e_i)(0, e_j) = (gamma e_j* e_i, 0)
        for k, value in enumerate(old[(j, i)]):
            products[(n + i, n + j)][k] += scale * signs[j] * value
    labels = ("1", *(f"e{index}" for index in range(1, 2 * n)))
    doubled_signs = (*signs, *(-1 for _ in range(n)))
    return _build(name, labels, products, doubled_signs)


def complexes() -> Algebra:
    """Return the complex numbers over the rationals, basis ``1, e1``."""
    return cayley_dickson(reals(), name="C")


def split_complexes() -> Algebra:
    """Return the split complex numbers, where ``e1*e1 = 1``."""
    return cayley_dickson(reals(), gamma=1, name="Cs")


def quaternions() -> Algebra:
    """Return the quaternions by doubling the complex numbers."""
    return cayley_dickson(complexes(), name="H")


def split_quaternions() -> Algebra:
    """Return the split quaternions by a split doubling of the complex numbers."""
    return cayley_dickson(complexes(), gamma=1, name="Hs")


def octonions() -> Algebra:
    """Return the octonions by doubling the quaternions."""
    return cayley_dickson(quaternions(), name="O")


def split_octonions() -> Algebra:
    """Return the split octonions by a split doubling of the quaternions."""
    return cayley_dickson(quaternions(), gamma=1, name="Os")


def sedenions() -> Algebra:
    """Return the sedenions, the first doubling that is not a composition algebra."""
    return cayley_dickson(octonions(), name="S")


def tensor(left: Algebra, right: Algebra, *, name: str | None = None) -> Algebra:
    """Return the tensor product of two algebras with factor-wise product.

    The basis is ordered with the right factor running fastest.  A basis
    element is labeled ``a.b``, with the unit factors left out, so the unit is
    ``1``.  Where the two factors share a label, the right factor's ``e1, e2,
    ...`` become ``f1, f2, ...``.  The declared conjugation conjugates both factors;
    :func:`factor_conjugation` gives the one-sided ones.
    """
    if left.unit_index is None or right.unit_index is None:
        raise EasyError("unit", "both factors need a declared unit")

    shared = (set(left.labels) & set(right.labels)) - {"1"}

    def second_name(b: str) -> str:
        if not shared:
            return b
        return "f" + b[1:] if b.startswith("e") and b[1:].isdigit() else b + "_2"

    def label(a: str, b: str) -> str:
        if a == "1":
            return b if b == "1" else second_name(b)
        return a if b == "1" else f"{a}.{second_name(b)}"

    pairs = list(product(range(left.rank), range(right.rank)))
    labels = [label(left.labels[i], right.labels[j]) for i, j in pairs]
    if len(set(labels)) != len(labels):
        raise EasyError(
            "table", "the factor labels do not give distinct product labels"
        )
    first, second = _table(left), _table(right)
    width = right.rank
    products: dict[tuple[int, int], list[Fraction]] = {}
    for (p, (i, j)), (q, (k, m)) in product(enumerate(pairs), repeat=2):
        cell = [Fraction(0)] * len(pairs)
        for a, x in enumerate(first[(i, k)]):
            if x != 0:
                for b, y in enumerate(second[(j, m)]):
                    if y != 0:
                        cell[a * width + b] += x * y
        products[(p, q)] = cell
    both = _factor_signs(left, right, first=True, second=True)
    return _build(name or f"{left.name}x{right.name}", labels, products, both)


def _factor_signs(
    left: Algebra, right: Algebra, *, first: bool, second: bool
) -> tuple[int, ...]:
    one = tuple(1 for _ in range(max(left.rank, right.rank)))
    a = left.conjugation_signs if first else one[: left.rank]
    b = right.conjugation_signs if second else one[: right.rank]
    if a is None or b is None:
        raise EasyError("involution", "a factor declares no conjugation")
    return tuple(x * y for x, y in product(a, b))


def factor_conjugation(
    left: Algebra, right: Algebra, *, first: bool = False, second: bool = False
) -> tuple[int, ...]:
    """Return the signs that conjugate the chosen factors of ``tensor(left, right)``."""
    return _factor_signs(left, right, first=first, second=second)


def conjugate(x: Element, signs: Sequence[int]) -> Element:
    """Apply a sign involution to an element."""
    if len(signs) != x.algebra.rank or any(sign not in (1, -1) for sign in signs):
        raise EasyError("involution", "one sign, +1 or -1, per basis element")
    return x.algebra(
        *(sign * value for sign, value in zip(signs, x.vector, strict=True))
    )


def _signs(source: Algebra, signs: Sequence[int] | None) -> tuple[int, ...]:
    chosen = source.conjugation_signs if signs is None else tuple(signs)
    if chosen is None:
        raise EasyError("involution", f"{source.name} declares no conjugation")
    return chosen


def is_involution(source: Algebra, signs: Sequence[int] | None = None) -> bool:
    """Decide exactly whether the signs give an anti-automorphism of order two.

    The condition ``(x y)* = y* x*`` is bilinear, so the basis grid decides it.
    """
    chosen = _signs(source, signs)
    return all(
        conjugate(a * b, chosen) == conjugate(b, chosen) * conjugate(a, chosen)
        for a, b in product(source.basis, repeat=2)
    )


def is_automorphism(source: Algebra, images: Sequence[Element]) -> bool:
    """Decide exactly whether basis images define an algebra automorphism.

    ``images[i]`` is the image of the i-th basis element.  The map must be
    invertible and satisfy ``f(x y) = f(x) f(y)``, which the basis grid decides.
    """
    if len(images) != source.rank or any(x.algebra is not source for x in images):
        raise EasyError(
            "coordinates", "one image in the same algebra per basis element"
        )

    def apply(x: Element) -> Element:
        total = source.zero
        for value, image in zip(x.vector, images, strict=True):
            if value != 0:
                total = total + image * value
        return total

    multiplicative = all(
        apply(a * b) == images[i] * images[j]
        for (i, a), (j, b) in product(enumerate(source.basis), repeat=2)
    )
    return multiplicative and _rank([list(x.vector) for x in images]) == source.rank


def _rank(rows: list[list[Fraction]]) -> int:
    rank = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][column] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r][column] != 0:
                factor = rows[r][column] / rows[rank][column]
                rows[r] = [
                    x - factor * y for x, y in zip(rows[r], rows[rank], strict=True)
                ]
        rank += 1
    return rank


def inner(x: Element, y: Element) -> Fraction:
    """Return the symmetric form ``(x y* + y x*) / 2`` as a rational number."""
    source = x.algebra
    value = (x * y.conj() + y * x.conj()) / 2
    unit = source.unit_index
    scalar = value.vector[unit] if unit is not None else Fraction(0)
    if unit is None or value != source.scalar(scalar):
        raise EasyError("norm", "x y* + y x* is not a multiple of the unit")
    return scalar


def is_composition(source: Algebra) -> bool:
    """Decide exactly whether the norm is multiplicative, ``N(xy) = N(x) N(y)``.

    The quartic identity is equivalent to its polarization
    ``<ac, bd> + <ad, bc> = 2 <a, b> <c, d>`` on all basis quadruples.
    """
    try:
        form = {
            (i, j): inner(a, b)
            for (i, a), (j, b) in product(enumerate(source.basis), repeat=2)
        }
    except EasyError:
        return False
    products = _table(source)

    def pair(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
        return sum(
            (
                x * y * form[(i, j)]
                for i, x in enumerate(u)
                if x
                for j, y in enumerate(v)
                if y
            ),
            Fraction(0),
        )

    indices = range(source.rank)
    return all(
        pair(products[(a, c)], products[(b, d)])
        + pair(products[(a, d)], products[(b, c)])
        == 2 * form[(a, b)] * form[(c, d)]
        for a, b, c, d in product(indices, repeat=4)
    )


def is_alternative(source: Algebra) -> bool:
    """Decide exactly whether ``(xx)y = x(xy)`` and ``(yx)x = y(xx)`` always hold.

    Both laws are quadratic in ``x``; their linearizations on the basis grid
    are equivalent to them over the rationals.
    """
    basis = source.basis
    return all(
        (a * b) * c + (b * a) * c == a * (b * c) + b * (a * c)
        and (c * a) * b + (c * b) * a == c * (a * b) + c * (b * a)
        for a, b, c in product(basis, repeat=3)
    )


def is_associative(source: Algebra) -> bool:
    """Decide exactly whether the product is associative."""
    return all(
        (a * b) * c == a * (b * c) for a, b, c in product(source.basis, repeat=3)
    )


def is_commutative(source: Algebra) -> bool:
    """Decide exactly whether the product is commutative."""
    return all(a * b == b * a for a, b in product(source.basis, repeat=2))


def left_divide(a: Element, b: Element) -> Element:
    """Return ``a^{-1} b``, which solves ``a x = b`` in an alternative algebra."""
    return a.inv() * b


def right_divide(b: Element, a: Element) -> Element:
    """Return ``b a^{-1}``, which solves ``x a = b`` in an alternative algebra."""
    return b * a.inv()
