"""Clifford and Grassmann algebras, and exact gamma matrices.

Two tools for one subject::

    import anyalgebra.clifford as ac

    Cl = ac.clifford(1, 3)  # the 16-dimensional algebra itself
    print(Cl.e1 * Cl.e2, Cl.e2 * Cl.e1)  # e12 -e12

    gammas = ac.gamma_matrices(11, 3)  # 128-by-128 matrices, built instantly
    print(ac.satisfies_clifford_relations(gammas, 11, 3))  # True

``clifford(p, q, r)`` returns a handle of :mod:`anyalgebra.easy` whose basis
is the blades ``1, e1, ..., e12, ...``, ordered by grade and then
lexicographically.  The generators ``e1 .. ep`` square to ``+1``, the next
``q`` to ``-1``, and the last ``r`` to ``0``.  The full multiplication table
has ``4^n`` cells, so this form is for ``n <= 8``.

Gamma matrices are kept as *monomial* matrices: one nonzero entry per row,
each a power of ``i``.  Products stay monomial, so every relation is checked
exactly and a 256-by-256 representation costs almost nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from itertools import combinations, product

import anyalgebra.composition as ca
import anyalgebra.matrices as am
from anyalgebra.easy import Algebra, EasyError, Element, algebra
from anyalgebra.lie import LieAlgebra, _Span, from_matrices

MAX_TABLE_GENERATORS = 8


# --- blades ------------------------------------------------------------------


def blade_product(left: int, right: int, signature: Sequence[int]) -> tuple[int, int]:
    """Multiply two basis blades given as bit masks.

    Bit ``k`` of a mask says that generator ``k + 1`` is a factor.  The result
    is ``(sign, mask)`` with ``sign`` in ``{-1, 0, 1}``; ``signature[k]`` is
    the square of generator ``k + 1``.
    """
    sign = 1
    shifted = left >> 1
    while shifted:
        if bin(shifted & right).count("1") % 2:
            sign = -sign
        shifted >>= 1
    common = left & right
    index = 0
    while common:
        if common & 1:
            sign *= signature[index]
        common >>= 1
        index += 1
    return sign, left ^ right


def _blades(n: int) -> list[int]:
    return sorted(range(1 << n), key=lambda mask: (bin(mask).count("1"), _digits(mask)))


def _digits(mask: int) -> tuple[int, ...]:
    return tuple(k + 1 for k in range(mask.bit_length()) if mask >> k & 1)


def _label(mask: int) -> str:
    return "1" if mask == 0 else "e" + "".join(str(d) for d in _digits(mask))


def clifford(p: int, q: int = 0, r: int = 0, *, name: str | None = None) -> Algebra:
    """Return the Clifford algebra ``Cl(p, q, r)`` over the rationals.

    The declared conjugation is reversion, so ``x.conj()`` reverses the order
    of the generators in every blade.
    """
    n = p + q + r
    if min(p, q, r) < 0 or not 1 <= n <= MAX_TABLE_GENERATORS:
        raise EasyError(
            "signature",
            f"the table form needs 1 to {MAX_TABLE_GENERATORS} generators; "
            "use gamma_matrices for larger signatures",
        )
    signature = [1] * p + [-1] * q + [0] * r
    blades = _blades(n)
    labels = tuple(_label(mask) for mask in blades)
    table: dict[tuple[str, str], str] = {}
    for a in blades:
        for b in blades:
            sign, mask = blade_product(a, b, signature)
            if sign:
                table[(_label(a), _label(b))] = ("-" if sign < 0 else "") + _label(mask)
    default = f"Cl({p},{q})" if r == 0 else f"Cl({p},{q},{r})"
    plain = algebra(labels, table, name=name or default)
    return Algebra(plain.structure, name=plain.name, conjugation_signs=reversion(plain))


def grassmann(n: int) -> Algebra:
    """Return the exterior algebra on ``n`` generators, ``Cl(0, 0, n)``."""
    return clifford(0, 0, n, name=f"Gr({n})")


def grades(source: Algebra) -> tuple[int, ...]:
    """Return the grade of every basis blade."""
    if any(
        label != "1" and not (label[0] == "e" and label[1:].isdigit())
        for label in source.labels
    ):
        raise EasyError("signature", f"{source.name} does not have a blade basis")
    return tuple(0 if label == "1" else len(label) - 1 for label in source.labels)


def grade_involution(source: Algebra) -> tuple[int, ...]:
    """Return the signs of the grade involution, ``(-1)^k`` on grade ``k``."""
    return tuple((-1) ** k for k in grades(source))


def reversion(source: Algebra) -> tuple[int, ...]:
    """Return the signs of reversion, ``(-1)^(k(k-1)/2)`` on grade ``k``."""
    return tuple((-1) ** (k * (k - 1) // 2) for k in grades(source))


def clifford_conjugation(source: Algebra) -> tuple[int, ...]:
    """Return the signs of Clifford conjugation, ``(-1)^(k(k+1)/2)`` on grade ``k``."""
    return tuple((-1) ** (k * (k + 1) // 2) for k in grades(source))


def grade_part(x: Element, k: int) -> Element:
    """Return the grade-``k`` part of a multivector."""
    found = grades(x.algebra)
    return x.algebra(
        *(v if g == k else 0 for v, g in zip(x.vector, found, strict=True))
    )


def pseudoscalar(source: Algebra) -> Element:
    """Return the product of all generators, the top-grade blade."""
    found = grades(source)
    return source.basis[found.index(max(found))]


def even_subalgebra(source: Algebra) -> Algebra:
    """Return the subalgebra spanned by the blades of even grade."""
    found = grades(source)
    keep = [index for index, k in enumerate(found) if k % 2 == 0]
    labels = tuple(source.labels[index] for index in keep)
    table = {}
    for i in keep:
        for j in keep:
            cell = (source.basis[i] * source.basis[j]).coefficients
            if cell:
                table[(source.labels[i], source.labels[j])] = cell
    plain = algebra(labels, table, name=f"{source.name}+")
    return Algebra(plain.structure, name=plain.name, conjugation_signs=reversion(plain))


# --- monomial matrices -------------------------------------------------------


class Monomial:
    """A square matrix with one nonzero entry per row, ``scale * i^k``.

    ``columns[r]`` is the column of the entry in row ``r`` and ``phases[r]``
    the power of ``i`` there.  ``scale`` is one rational factor for the whole
    matrix.
    """

    __slots__ = ("columns", "phases", "scale")
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, columns: Sequence[int], phases: Sequence[int], scale: int | Fraction = 1
    ) -> None:
        if sorted(columns) != list(range(len(columns))) or len(phases) != len(columns):
            raise EasyError("shape", "one entry per row and per column is required")
        self.columns = tuple(columns)
        self.phases = tuple(phase % 4 for phase in phases)
        self.scale = Fraction(scale)

    @property
    def size(self) -> int:
        """Return the number of rows."""
        return len(self.columns)

    def __matmul__(self, other: Monomial) -> Monomial:
        if other.size != self.size:
            raise EasyError("shape", "the matrices have different sizes")
        return Monomial(
            [other.columns[c] for c in self.columns],
            [
                k + other.phases[c]
                for c, k in zip(self.columns, self.phases, strict=True)
            ],
            self.scale * other.scale,
        )

    def __mul__(self, scalar: int | Fraction) -> Monomial:
        return Monomial(self.columns, self.phases, self.scale * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> Monomial:
        return self * -1

    def times_i(self, power: int = 1) -> Monomial:
        """Return the matrix multiplied by ``i^power``."""
        return Monomial(self.columns, [k + power for k in self.phases], self.scale)

    def transpose(self) -> Monomial:
        """Return the transpose."""
        columns = [0] * self.size
        phases = [0] * self.size
        for row, (column, phase) in enumerate(
            zip(self.columns, self.phases, strict=True)
        ):
            columns[column], phases[column] = row, phase
        return Monomial(columns, phases, self.scale)

    def dagger(self) -> Monomial:
        """Return the conjugate transpose."""
        transposed = self.transpose()
        return Monomial(transposed.columns, [-k for k in transposed.phases], self.scale)

    def _normal(self) -> tuple[tuple[int, ...], tuple[int, ...], Fraction]:
        if self.scale < 0:
            return self.columns, tuple((k + 2) % 4 for k in self.phases), -self.scale
        return self.columns, self.phases, self.scale

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Monomial):
            return NotImplemented
        if self.scale == 0 or other.scale == 0:
            return self.scale == other.scale and self.size == other.size
        return self._normal() == other._normal()

    def is_identity_multiple(self) -> tuple[int, Fraction] | None:
        """Return ``(k, s)`` when the matrix is ``s * i^k`` times the identity."""
        columns, phases, scale = self._normal()
        if columns != tuple(range(self.size)) or len(set(phases)) != 1:
            return None
        return phases[0], scale

    def dense(self) -> tuple[tuple[tuple[Fraction, Fraction], ...], ...]:
        """Return the entries as ``(real, imaginary)`` pairs."""
        unit = ((1, 0), (0, 1), (-1, 0), (0, -1))
        zero = (Fraction(0), Fraction(0))
        rows = []
        for column, phase in zip(self.columns, self.phases, strict=True):
            row = [zero] * self.size
            real, imaginary = unit[phase]
            row[column] = (real * self.scale, imaginary * self.scale)
            rows.append(tuple(row))
        return tuple(rows)

    def realified(self) -> tuple[tuple[Fraction, ...], ...]:
        """Return the real block matrix ``[[A, -B], [B, A]]`` of ``A + iB``."""
        dense = self.dense()
        top = [[z[0] for z in row] + [-z[1] for z in row] for row in dense]
        bottom = [[z[1] for z in row] + [z[0] for z in row] for row in dense]
        return tuple(tuple(row) for row in top + bottom)

    def __repr__(self) -> str:
        return f"Monomial(size={self.size})"


def identity(size: int) -> Monomial:
    """Return the identity matrix."""
    return Monomial(range(size), [0] * size)


def kron(left: Monomial, right: Monomial) -> Monomial:
    """Return the Kronecker product of two monomial matrices."""
    columns, phases = [], []
    for a, p in zip(left.columns, left.phases, strict=True):
        for b, q in zip(right.columns, right.phases, strict=True):
            columns.append(a * right.size + b)
            phases.append(p + q)
    return Monomial(columns, phases, left.scale * right.scale)


def pauli() -> tuple[Monomial, Monomial, Monomial]:
    """Return the Pauli matrices ``sigma_1, sigma_2, sigma_3``."""
    return (
        Monomial((1, 0), (0, 0)),
        Monomial((1, 0), (3, 1)),
        Monomial((0, 1), (0, 2)),
    )


def gamma_matrices(p: int, q: int = 0) -> tuple[Monomial, ...]:
    """Return gamma matrices for signature ``(p, q)``.

    The first ``p`` matrices square to ``+1`` and the next ``q`` to ``-1``;
    distinct ones anticommute.  They have size ``2^floor((p+q)/2)``, which is
    the irreducible complex dimension.  The construction is the standard
    tensor product of Pauli matrices: generators ``2k-1`` and ``2k`` carry
    ``sigma_1`` and ``sigma_2`` in slot ``k`` behind ``sigma_3`` factors, and
    an odd last generator is ``sigma_3`` in every slot.
    """
    n = p + q
    if n < 1 or p < 0 or q < 0:
        raise EasyError("signature", "at least one generator is required")
    slots = max(n // 2, 1)
    s1, s2, s3 = pauli()
    one = identity(2)

    def chain(position: int, middle: Monomial) -> Monomial:
        factors = [s3] * position + [middle] + [one] * (slots - position - 1)
        result = factors[0]
        for factor in factors[1:]:
            result = kron(result, factor)
        return result

    euclidean = []
    for k in range(n // 2):
        euclidean += [chain(k, s1), chain(k, s2)]
    if n == 1:
        euclidean.append(s1)
    elif n % 2:
        last = s3
        for _ in range(slots - 1):
            last = kron(last, s3)
        euclidean.append(last)
    return tuple(g if index < p else g.times_i() for index, g in enumerate(euclidean))


def satisfies_clifford_relations(
    gammas: Sequence[Monomial], p: int, q: int = 0
) -> bool:
    """Check ``g_a g_b + g_b g_a = 2 eta_ab`` exactly for every pair."""
    if len(gammas) != p + q:
        return False
    for a, gamma in enumerate(gammas):
        square = (gamma @ gamma).is_identity_multiple()
        if square != ((0 if a < p else 2), Fraction(1)):
            return False
    return all(a @ b == -(b @ a) for a, b in combinations(gammas, 2))


def chirality(gammas: Sequence[Monomial]) -> Monomial:
    """Return the product of all gamma matrices, rescaled to square to ``+1``."""
    product = gammas[0]
    for gamma in gammas[1:]:
        product = product @ gamma
    found = (product @ product).is_identity_multiple()
    if found is None:  # pragma: no cover - a product of gammas squares to a phase
        raise EasyError("signature", "the product does not square to a scalar")
    return product if found[0] == 0 else product.times_i()


def charge_conjugations(gammas: Sequence[Monomial]) -> dict[int, Monomial]:
    """Return matrices ``C`` with ``C g C^-1 = sign * g^T`` for every gamma.

    The result maps each available ``sign`` (``+1`` or ``-1``) to one such
    matrix.  Even dimensions have both; odd dimensions have one.
    """
    symmetric = [g for g in gammas if g.transpose() == g]
    skew = [g for g in gammas if g.transpose() == -g]
    found: dict[int, Monomial] = {}
    for family in (symmetric, skew):
        candidate = identity(gammas[0].size)
        for gamma in family:
            candidate = candidate @ gamma
        inverse = candidate.dagger() * (1 / (candidate.scale * candidate.scale))
        for sign in (1, -1):
            if all(candidate @ g @ inverse == sign * g.transpose() for g in gammas):
                found.setdefault(sign, candidate)
    return found


def spin_generators(gammas: Sequence[Monomial]) -> tuple[Monomial, ...]:
    """Return ``g_a g_b / 2`` for ``a < b``, which span the spin Lie algebra."""
    return tuple((a @ b) * Fraction(1, 2) for a, b in combinations(gammas, 2))


def spin_algebra(p: int, q: int = 0) -> LieAlgebra:
    """Return ``spin(p, q)`` from the commutators of gamma-matrix products."""
    generators = spin_generators(gamma_matrices(p, q))
    name = f"spin({p})" if q == 0 else f"spin({p},{q})"
    return from_matrices([g.realified() for g in generators], name=name)


# --- idempotents and minimal left ideals -------------------------------------


def commuting_involutions(source: Algebra) -> tuple[Element, ...]:
    """Return a maximal set of independent commuting blades that square to ``+1``.

    Blades are taken in basis order.  A blade is skipped when it does not
    commute with the chosen ones or is, up to sign, a product of them.  These
    are the blades that can be made diagonal together.
    """
    grades(source)
    one = source.unit
    assert one is not None
    chosen: list[Element] = []
    generated = {source.labels[0]}
    for blade in source.basis[1:]:
        (label,) = blade.coefficients
        if label in generated or blade * blade != one:
            continue
        if any(blade * other != other * blade for other in chosen):
            continue
        chosen.append(blade)
        generated |= {
            next(iter((blade * source[known]).coefficients)) for known in generated
        }
    return tuple(chosen)


def idempotent(
    blades: Sequence[Element], signs: Sequence[int] | None = None
) -> Element:
    """Return the product of the factors ``(1 + s*b) / 2`` over the given blades."""
    if not blades:
        raise EasyError("shape", "at least one blade is required")
    chosen = (1,) * len(blades) if signs is None else tuple(signs)
    if len(chosen) != len(blades) or any(sign not in (1, -1) for sign in chosen):
        raise EasyError("shape", "one sign, +1 or -1, per blade")
    result = (1 + chosen[0] * blades[0]) / 2
    for sign, blade in zip(chosen[1:], blades[1:], strict=True):
        result = result * ((1 + sign * blade) / 2)
    return result


def is_idempotent(x: Element) -> bool:
    """Say whether ``x * x == x``."""
    return x * x == x


def primitive_idempotents(source: Algebra) -> tuple[Element, ...]:
    """Return mutually annihilating primitive idempotents that sum to one.

    There is one for each choice of signs on :func:`commuting_involutions`.
    An algebra with no such blade, such as ``Cl(0, 1)``, has only ``1``.
    """
    blades = commuting_involutions(source)
    if not blades:
        one = source.unit
        assert one is not None
        return (one,)
    return tuple(
        idempotent(blades, signs) for signs in product((1, -1), repeat=len(blades))
    )


def _span_of(vectors: Sequence[Element]) -> tuple[_Span, tuple[Element, ...]]:
    span = _Span(vectors[0].algebra.rank)
    kept = tuple(vector for vector in vectors if span.add(vector.vector))
    return span, kept


def left_ideal(f: Element) -> tuple[Element, ...]:
    """Return a basis of the left ideal ``A f``."""
    return _span_of([blade * f for blade in f.algebra.basis])[1]


def division_ring(f: Element) -> str:
    """Return ``R``, ``C`` or ``H``: the algebra ``f A f`` of a primitive idempotent.

    The answer is read off the dimension of ``f A f``, which must be 1, 2 or
    4; any other dimension means that ``f`` is not primitive.
    """
    if not is_idempotent(f) or not f:
        raise EasyError("idempotent", "a nonzero idempotent is required")
    corner = _span_of([f * blade * f for blade in f.algebra.basis])[1]
    found = {1: "R", 2: "C", 4: "H"}.get(len(corner))
    if found is None:
        raise EasyError(
            "idempotent", f"f A f has dimension {len(corner)}: f is not primitive"
        )
    return found


def spinor_representation(
    source: Algebra, f: Element | None = None
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    """Return real matrices of the generators acting on a minimal left ideal.

    The ideal is ``A f`` for a primitive idempotent ``f``, by default the
    first of :func:`primitive_idempotents`.  Matrix ``a`` gives left
    multiplication by generator ``a + 1`` in the basis of
    :func:`left_ideal`.  The matrices are rational and satisfy the Clifford
    relations of ``source``.  When :func:`division_ring` is ``R`` this is a
    Majorana representation.
    """
    chosen = primitive_idempotents(source)[0] if f is None else f
    span, basis = _span_of([blade * chosen for blade in source.basis])
    generators = [
        blade
        for blade, grade in zip(source.basis, grades(source), strict=True)
        if grade == 1
    ]
    matrices = []
    for generator in generators:
        columns = []
        for vector in basis:
            found = span.coordinates((generator * vector).vector)
            assert found is not None
            columns.append(found)
        matrices.append(
            tuple(
                tuple(columns[c][r] for c in range(len(basis)))
                for r in range(len(basis))
            )
        )
    return tuple(matrices)


# --- dense complex matrices: changes of basis and vectors --------------------


@cache
def complex_numbers() -> Algebra:
    """Return the one complex algebra that dense gamma matrices share."""
    return ca.complexes()


def as_matrix(value: Monomial) -> am.Mat:
    """Return a monomial matrix as a dense matrix over the complex numbers."""
    numbers = complex_numbers()
    return am.matrix(
        numbers,
        [
            [numbers(real, imaginary) for real, imaginary in row]
            for row in value.dense()
        ],
    )


def change_basis(
    gammas: Sequence[am.Mat], s: am.Mat, s_inverse: am.Mat
) -> tuple[am.Mat, ...]:
    """Return ``S g S^-1`` for every matrix, after checking ``S S^-1 = 1``."""
    size = s.shape[0]
    if s @ s_inverse != am.identity(s.algebra, size):
        raise EasyError("shape", "the second matrix is not the inverse of the first")
    return tuple(s @ gamma @ s_inverse for gamma in gammas)


def dirac_basis(
    gammas: Sequence[Monomial], index: int = 0
) -> tuple[tuple[am.Mat, ...], am.Mat]:
    """Return the gamma matrices in a basis where one of them is diagonal.

    The construction above is a Weyl basis: its chirality matrix is diagonal.
    Here gamma ``index`` becomes diagonal and the chirality matrix does not.
    The result is ``(matrices, S)`` with ``S = c*g + G``, where ``G`` is the
    chirality matrix and ``c`` is 1 or ``i`` so that ``(c*g)^2 = 1``.  ``S``
    is its own inverse up to a factor 2, so applying the change twice
    returns the Weyl basis.  An even number of gamma matrices is required.
    """
    if len(gammas) % 2:
        raise EasyError("signature", "an even number of gamma matrices is required")
    chosen = gammas[index]
    square = (chosen @ chosen).is_identity_multiple()
    assert square is not None
    unitary = chosen if square[0] == 0 else chosen.times_i()
    s = as_matrix(unitary) + as_matrix(chirality(gammas))
    return change_basis([as_matrix(g) for g in gammas], s, s / 2), s


def is_real(matrices: Sequence[am.Mat]) -> bool:
    """Say whether every entry of every matrix is real."""
    return all(matrix.conj() == matrix for matrix in matrices)


def is_imaginary(matrices: Sequence[am.Mat]) -> bool:
    """Say whether every entry of every matrix is imaginary."""
    return all(matrix.conj() == -matrix for matrix in matrices)


def majorana_matrices(
    p: int, q: int = 0
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    """Return real gamma matrices of the smallest size for signature ``(p, q)``.

    They exist in the complex dimension ``2^floor((p+q)/2)`` exactly when the
    minimal left ideal of ``Cl(p, q)`` is a real vector space with division
    ring ``R``; otherwise the error names the ring.
    """
    source = clifford(p, q)
    ring = division_ring(primitive_idempotents(source)[0])
    if ring != "R":
        raise EasyError(
            "signature",
            f"Cl({p},{q}) has spinors over {ring}: no real representation of "
            "the complex dimension exists",
        )
    return spinor_representation(source)


def slash(gammas: Sequence[Monomial], vector: Sequence[int | Fraction]) -> am.Mat:
    """Return the matrix ``v_a g_a`` of a vector."""
    if len(vector) != len(gammas):
        raise EasyError("shape", "one component per gamma matrix is required")
    total = am.zeros(complex_numbers(), gammas[0].size)
    for component, gamma in zip(vector, gammas, strict=True):
        total = total + as_matrix(gamma) * Fraction(component)
    return total


def unslash(gammas: Sequence[Monomial], matrix: am.Mat) -> tuple[Fraction, ...]:
    """Return the vector ``v`` with ``slash(gammas, v) == matrix``.

    The components are ``tr(g_a M) / tr(g_a g_a)``.  A matrix that is not of
    that form is an error.
    """
    size = gammas[0].size
    components = []
    for gamma in gammas:
        square = (gamma @ gamma).is_identity_multiple()
        assert square is not None
        sign = 1 if square[0] == 0 else -1
        trace = (as_matrix(gamma) @ matrix).trace()
        components.append(trace.vector[0] / (sign * size))
    if slash(gammas, components) != matrix:
        raise EasyError("shape", "the matrix is not a combination of the gammas")
    return tuple(components)
