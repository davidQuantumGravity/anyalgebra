"""Lie algebras from structure constants or matrices, and root systems.

Everything is exact rational arithmetic::

    import anyalgebra.lie as al

    g = al.so(3, 1)
    print(g.dimension, g.jacobi_holds(), g.killing_signature())  # 6 True (3, 3, 0)
    print(al.root_system("F", 4).count)  # 48

A :class:`LieAlgebra` stores a basis and its brackets.  It can be built from
structure constants, or from matrices whose commutators close, in which case
the structure constants are recovered by exact linear algebra and the closure
is checked rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations, product

from anyalgebra.easy import Algebra, EasyError, algebra

Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Rows = Sequence[Sequence[int | Fraction]]


# --- exact linear algebra ----------------------------------------------------


class _Span:
    """An incrementally built basis that can decompose vectors exactly."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.size = 0
        # reduced rows: pivot column -> (vector, combination of the added vectors)
        self._rows: dict[int, tuple[list[Fraction], list[Fraction]]] = {}

    def _reduce(
        self, vector: Sequence[Fraction]
    ) -> tuple[list[Fraction], list[Fraction]]:
        rest = list(vector)
        used = [Fraction(0)] * self.size
        for pivot, (row, combination) in self._rows.items():
            factor = rest[pivot]
            if factor:
                for index, value in enumerate(row):
                    if value:
                        rest[index] -= factor * value
                for index, value in enumerate(combination):
                    used[index] += factor * value
        return rest, used

    def add(self, vector: Sequence[Fraction]) -> bool:
        """Add a vector; return False when it already lies in the span."""
        rest, used = self._reduce(vector)
        pivot = next((index for index, value in enumerate(rest) if value), None)
        if pivot is None:
            return False
        scale = rest[pivot]
        combination = [-value / scale for value in used] + [1 / scale]
        row = [value / scale for value in rest]
        for other, (other_row, other_combination) in self._rows.items():
            factor = other_row[pivot]
            other_combination.append(Fraction(0))
            if factor:
                self._rows[other] = (
                    [a - factor * b for a, b in zip(other_row, row, strict=True)],
                    [
                        a - factor * b
                        for a, b in zip(other_combination, combination, strict=True)
                    ],
                )
        self._rows[pivot] = (row, combination)
        self.size += 1
        return True

    def coordinates(self, vector: Sequence[Fraction]) -> Vector | None:
        """Return the coordinates in the added vectors, or None outside the span."""
        rest, used = self._reduce(vector)
        return None if any(rest) else tuple(used)


def _exact(rows: Rows) -> Matrix:
    matrix = tuple(tuple(Fraction(value) for value in row) for row in rows)
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise EasyError("shape", "a square matrix is required")
    return matrix


def _multiply(a: Matrix, b: Matrix) -> Matrix:
    size = len(a)
    columns = [[b[k][j] for k in range(size)] for j in range(size)]
    return tuple(
        tuple(
            sum(
                (x * y for x, y in zip(row, column, strict=True) if x and y),
                Fraction(0),
            )
            for column in columns
        )
        for row in a
    )


def _commutator(a: Matrix, b: Matrix) -> Matrix:
    ab, ba = _multiply(a, b), _multiply(b, a)
    return tuple(
        tuple(x - y for x, y in zip(r, s, strict=True))
        for r, s in zip(ab, ba, strict=True)
    )


def _flat(matrix: Matrix) -> Vector:
    return tuple(value for row in matrix for value in row)


def inertia(form: Sequence[Sequence[Fraction]]) -> tuple[int, int, int]:
    """Return the numbers of positive, negative and zero squares of a symmetric form.

    The form is diagonalized by exact congruence, so the answer is Sylvester's
    inertia and does not depend on floating point.
    """
    rows = [list(row) for row in form]
    size = len(rows)
    positive = negative = 0
    for start in range(size):
        pivot = next((i for i in range(start, size) if rows[i][i]), None)
        if pivot is None:
            found = next(
                (
                    (i, j)
                    for i in range(start, size)
                    for j in range(i + 1, size)
                    if rows[i][j]
                ),
                None,
            )
            if found is None:
                break
            i, j = found
            for k in range(size):
                rows[i][k] += rows[j][k]
            for k in range(size):
                rows[k][i] += rows[k][j]
            pivot = i
        if pivot != start:
            rows[start], rows[pivot] = rows[pivot], rows[start]
            for row in rows:
                row[start], row[pivot] = row[pivot], row[start]
        diagonal = rows[start][start]
        positive += diagonal > 0
        negative += diagonal < 0
        for i in range(start + 1, size):
            factor = rows[i][start] / diagonal
            if factor:
                for k in range(size):
                    rows[i][k] -= factor * rows[start][k]
                for k in range(size):
                    rows[k][i] -= factor * rows[k][start]
    return positive, negative, size - positive - negative


# --- Lie algebras ------------------------------------------------------------


class LieAlgebra:
    """A finite-dimensional rational Lie algebra with a chosen basis."""

    def __init__(
        self,
        brackets: Mapping[tuple[int, int], Sequence[int | Fraction]],
        dimension: int,
        *,
        name: str = "g",
        labels: Sequence[str] | None = None,
        matrices: Sequence[Matrix] | None = None,
    ) -> None:
        self.dimension = dimension
        self.name = name
        self.labels = (
            tuple(labels) if labels else tuple(f"x{n}" for n in range(1, dimension + 1))
        )
        if len(self.labels) != dimension:
            raise EasyError("shape", "one label per basis element is required")
        zero = (Fraction(0),) * dimension
        self._table: list[list[Vector]] = [[zero] * dimension for _ in range(dimension)]
        for (i, j), value in brackets.items():
            vector = tuple(Fraction(entry) for entry in value)
            if len(vector) != dimension or not (
                0 <= i < dimension and 0 <= j < dimension
            ):
                raise EasyError(
                    "shape", f"the bracket ({i}, {j}) does not fit the dimension"
                )
            self._table[i][j] = vector
        for i, j in product(range(dimension), repeat=2):
            if self._table[i][j] != tuple(-value for value in self._table[j][i]):
                raise EasyError(
                    "bracket", f"the bracket is not antisymmetric at ({i}, {j})"
                )
        self.matrices = None if matrices is None else tuple(matrices)

    # brackets

    def basis_bracket(self, i: int, j: int) -> Vector:
        """Return ``[x_i, x_j]`` as coordinates."""
        return self._table[i][j]

    def bracket(
        self, u: Sequence[int | Fraction], v: Sequence[int | Fraction]
    ) -> Vector:
        """Return the bracket of two coordinate vectors."""
        total = [Fraction(0)] * self.dimension
        for i, a in enumerate(u):
            if a:
                for j, b in enumerate(v):
                    if b:
                        for k, c in enumerate(self._table[i][j]):
                            if c:
                                total[k] += a * b * c
        return tuple(total)

    def jacobi_holds(self) -> bool:
        """Decide the Jacobi identity on every basis triple."""
        unit = [
            tuple(Fraction(int(i == k)) for k in range(self.dimension))
            for i in range(self.dimension)
        ]
        for i, j, k in combinations(range(self.dimension), 3):
            terms = (
                self.bracket(unit[i], self._table[j][k]),
                self.bracket(unit[j], self._table[k][i]),
                self.bracket(unit[k], self._table[i][j]),
            )
            if any(a + b + c for a, b, c in zip(*terms, strict=True)):
                return False
        return True

    # invariants

    def adjoint(self, i: int) -> Matrix:
        """Return the matrix of ``ad x_i`` acting on column coordinates."""
        return tuple(
            tuple(self._table[i][j][k] for j in range(self.dimension))
            for k in range(self.dimension)
        )

    def killing_form(self) -> Matrix:
        """Return the Killing form ``trace(ad x_i ad x_j)`` in the basis."""
        size = self.dimension
        adjoints = [self.adjoint(i) for i in range(size)]
        transposed = [list(zip(*matrix, strict=True)) for matrix in adjoints]
        form = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                value = sum(
                    (
                        x * y
                        for row, column in zip(adjoints[i], transposed[j], strict=True)
                        for x, y in zip(row, column, strict=True)
                        if x and y
                    ),
                    Fraction(0),
                )
                form[i][j] = form[j][i] = value
        return tuple(tuple(row) for row in form)

    def killing_signature(self) -> tuple[int, int, int]:
        """Return the inertia ``(positive, negative, zero)`` of the Killing form.

        A semisimple algebra has no zeros (Cartan's criterion); the positive
        count is then the dimension of the noncompact part, and a compact form
        has none.
        """
        return inertia(self.killing_form())

    def is_semisimple(self) -> bool:
        """Decide semisimplicity by Cartan's criterion."""
        return self.dimension > 0 and self.killing_signature()[2] == 0

    def derived_dimension(self) -> int:
        """Return the dimension of ``[g, g]``."""
        span = _Span(self.dimension)
        for i, j in combinations(range(self.dimension), 2):
            span.add(self._table[i][j])
        return span.size

    def center_dimension(self) -> int:
        """Return the dimension of the center."""
        span = _Span(self.dimension)
        rows = [
            [self._table[i][j][k] for i in range(self.dimension)]
            for j in range(self.dimension)
            for k in range(self.dimension)
        ]
        for row in rows:
            span.add(row)
        return self.dimension - span.size

    def as_algebra(self) -> Algebra:
        """Return the bracket as a convenience-layer algebra without a unit."""
        table = {
            (self.labels[i], self.labels[j]): {
                self.labels[k]: value
                for k, value in enumerate(self._table[i][j])
                if value
            }
            for i, j in product(range(self.dimension), repeat=2)
            if any(self._table[i][j])
        }
        return algebra(self.labels, table, name=self.name, unit=None)

    def __repr__(self) -> str:
        return f"LieAlgebra({self.name!r}, dimension={self.dimension})"


def lie_closure(generators: Iterable[Rows], *, limit: int = 400) -> tuple[Matrix, ...]:
    """Return a basis of the Lie algebra generated by square rational matrices."""
    matrices = [_exact(rows) for rows in generators]
    if not matrices:
        raise EasyError("shape", "at least one generator is required")
    span = _Span(len(matrices[0]) ** 2)
    basis: list[Matrix] = []
    queue = list(matrices)
    while queue:
        candidate = queue.pop(0)
        if len(candidate) != len(matrices[0]):
            raise EasyError("shape", "the generators have different sizes")
        if span.add(_flat(candidate)):
            if len(basis) >= limit:
                raise EasyError("bound", f"the closure exceeds {limit} dimensions")
            queue.extend(_commutator(candidate, other) for other in basis)
            basis.append(candidate)
    return tuple(basis)


def from_matrices(
    matrices: Iterable[Rows], *, name: str = "g", labels: Sequence[str] | None = None
) -> LieAlgebra:
    """Build a Lie algebra from linearly independent matrices that close.

    The structure constants are solved exactly.  A dependent family or a
    commutator outside the span is an error that names the offending pair.
    """
    basis = [_exact(rows) for rows in matrices]
    if not basis or any(len(matrix) != len(basis[0]) for matrix in basis):
        raise EasyError("shape", "equally sized square matrices are required")
    span = _Span(len(basis[0]) ** 2)
    for index, matrix in enumerate(basis):
        if not span.add(_flat(matrix)):
            raise EasyError("dependent", f"matrix {index} depends on the earlier ones")
    brackets: dict[tuple[int, int], Vector] = {}
    for i, j in combinations(range(len(basis)), 2):
        found = span.coordinates(_flat(_commutator(basis[i], basis[j])))
        if found is None:
            raise EasyError(
                "closure", f"the commutator of matrices {i} and {j} leaves the span"
            )
        brackets[(i, j)] = found
        brackets[(j, i)] = tuple(-value for value in found)
    return LieAlgebra(brackets, len(basis), name=name, labels=labels, matrices=basis)


def generated_by(generators: Iterable[Rows], *, name: str = "g") -> LieAlgebra:
    """Return the Lie algebra generated by matrices under the commutator."""
    return from_matrices(lie_closure(generators), name=name)


# --- classical families ------------------------------------------------------


def _unit(size: int, row: int, column: int, value: int = 1) -> list[list[Fraction]]:
    matrix = [[Fraction(0)] * size for _ in range(size)]
    matrix[row][column] = Fraction(value)
    return matrix


def _add(
    a: list[list[Fraction]], b: list[list[Fraction]], sign: int = 1
) -> list[list[Fraction]]:
    return [
        [x + sign * y for x, y in zip(r, s, strict=True)]
        for r, s in zip(a, b, strict=True)
    ]


def gl(n: int) -> LieAlgebra:
    """Return ``gl(n)`` in the basis of matrix units."""
    units = [_unit(n, i, j) for i in range(n) for j in range(n)]
    labels = [f"E{i + 1}{j + 1}" for i in range(n) for j in range(n)]
    return from_matrices(units, name=f"gl({n})", labels=labels)


def sl(n: int) -> LieAlgebra:
    """Return the traceless matrices ``sl(n)``."""
    if n < 2:
        raise EasyError("shape", "sl(n) needs n >= 2")
    off = [_unit(n, i, j) for i in range(n) for j in range(n) if i != j]
    diagonal = [_add(_unit(n, i, i), _unit(n, i + 1, i + 1), -1) for i in range(n - 1)]
    return from_matrices(off + diagonal, name=f"sl({n})")


def so(p: int, q: int = 0) -> LieAlgebra:
    """Return ``so(p, q)``: matrices ``X`` with ``X^T eta + eta X = 0``.

    ``eta`` is diagonal with ``p`` entries ``+1`` followed by ``q`` entries
    ``-1``.  The basis is ``L_ab`` for ``a < b``.
    """
    n = p + q
    if n < 2 or p < 0 or q < 0:
        raise EasyError("shape", "so(p, q) needs p + q >= 2")
    eta = [1] * p + [-1] * q
    generators = [
        _add(_unit(n, a, b, eta[b]), _unit(n, b, a, eta[a]), -1)
        for a in range(n)
        for b in range(a + 1, n)
    ]
    labels = [f"L{a + 1}_{b + 1}" for a in range(n) for b in range(a + 1, n)]
    name = f"so({p})" if q == 0 else f"so({p},{q})"
    return from_matrices(generators, name=name, labels=labels)


def sp(n: int) -> LieAlgebra:
    """Return the real symplectic algebra ``sp(2n, R)`` of ``2n``-by-``2n`` matrices."""
    if n < 1:
        raise EasyError("shape", "sp(n) needs n >= 1")
    size = 2 * n
    generators = []
    for i in range(n):
        for j in range(n):
            generators.append(_add(_unit(size, i, j), _unit(size, n + j, n + i), -1))
    for i in range(n):
        for j in range(i, n):
            upper = _unit(size, i, n + j)
            lower = _unit(size, n + i, j)
            if i != j:
                upper = _add(upper, _unit(size, j, n + i))
                lower = _add(lower, _unit(size, n + j, i))
            generators += [upper, lower]
    return from_matrices(generators, name=f"sp({size},R)")


def su(p: int, q: int = 0) -> LieAlgebra:
    """Return ``su(p, q)`` as real matrices of size ``2(p+q)``.

    A complex matrix ``A + iB`` is written as the real block matrix
    ``[[A, -B], [B, A]]``.  The algebra consists of the traceless ``X`` with
    ``X^dagger eta + eta X = 0``.
    """
    n = p + q
    if n < 2 or p < 0 or q < 0:
        raise EasyError("shape", "su(p, q) needs p + q >= 2")
    eta = [1] * p + [-1] * q

    def real(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
        top = [ra + [-value for value in rb] for ra, rb in zip(a, b, strict=True)]
        bottom = [rb + ra for ra, rb in zip(a, b, strict=True)]
        return top + bottom

    zero = [[Fraction(0)] * n for _ in range(n)]
    generators = []
    for a in range(n):
        for b in range(a + 1, n):
            sign = eta[a] * eta[b]
            generators.append(
                real(_add(_unit(n, a, b), _unit(n, b, a, sign), -1), zero)
            )
            generators.append(real(zero, _add(_unit(n, a, b), _unit(n, b, a, sign))))
    for a in range(n - 1):
        generators.append(real(zero, _add(_unit(n, a, a), _unit(n, a + 1, a + 1), -1)))
    name = f"su({p})" if q == 0 else f"su({p},{q})"
    return from_matrices(generators, name=name)


# --- root systems ------------------------------------------------------------


def cartan_matrix(family: str, rank: int) -> tuple[tuple[int, ...], ...]:
    """Return the Cartan matrix of a simple type in the Bourbaki numbering."""
    valid = {
        "A": rank >= 1,
        "B": rank >= 2,
        "C": rank >= 2,
        "D": rank >= 4,
        "E": rank in (6, 7, 8),
        "F": rank == 4,
        "G": rank == 2,
    }
    if not valid.get(family, False):
        raise EasyError("type", f"{family}{rank} is not a simple Cartan type")
    matrix = [[2 if i == j else 0 for j in range(rank)] for i in range(rank)]

    def link(i: int, j: int, forward: int = -1, backward: int = -1) -> None:
        matrix[i][j], matrix[j][i] = forward, backward

    if family in "ABCFG":
        for i in range(rank - 1):
            link(i, i + 1)
    if family == "B":
        link(rank - 2, rank - 1, -2, -1)
    if family == "C":
        link(rank - 2, rank - 1, -1, -2)
    if family == "F":
        link(1, 2, -2, -1)
    if family == "G":
        link(0, 1, -1, -3)
    if family == "D":
        for i in range(rank - 2):
            link(i, i + 1)
        link(rank - 3, rank - 1)
    if family == "E":
        link(0, 2)
        link(1, 3)
        for i in range(2, rank - 1):
            link(i, i + 1)
    return tuple(tuple(row) for row in matrix)


class RootSystem:
    """The roots of a simple type, in coordinates of the simple roots."""

    def __init__(self, family: str, rank: int) -> None:
        self.family, self.rank = family, rank
        self.cartan = cartan_matrix(family, rank)
        self.positive = self._positive_roots()
        self.count = 2 * len(self.positive)
        self.dimension = self.count + rank

    def _positive_roots(self) -> tuple[tuple[int, ...], ...]:
        rank, cartan = self.rank, self.cartan
        simple = [tuple(int(i == j) for j in range(rank)) for i in range(rank)]
        found = set(simple)
        layer = list(simple)
        while layer:
            following = []
            for root in layer:
                for i in range(rank):
                    # <root, alpha_i^vee> in simple-root coordinates
                    pairing = sum(root[j] * cartan[j][i] for j in range(rank))
                    lowered = 0
                    probe = list(root)
                    while True:
                        probe[i] -= 1
                        if tuple(probe) in found:
                            lowered += 1
                        else:
                            break
                    if lowered - pairing > 0:
                        raised = tuple(v + int(j == i) for j, v in enumerate(root))
                        if raised not in found:
                            found.add(raised)
                            following.append(raised)
            layer = following
        return tuple(sorted(found, key=lambda root: (sum(root), root)))

    def _symmetrizer(self) -> tuple[Fraction, ...]:
        """Return squared root lengths ``d_i``, for which ``a_ij d_j`` is symmetric."""
        lengths: list[Fraction | None] = [None] * self.rank
        lengths[0] = Fraction(1)
        pending = [0]
        while pending:
            i = pending.pop()
            for j in range(self.rank):
                if lengths[j] is None and self.cartan[i][j]:
                    current = lengths[i]
                    assert current is not None
                    lengths[j] = current * Fraction(
                        self.cartan[j][i], self.cartan[i][j]
                    )
                    pending.append(j)
        return tuple(value for value in lengths if value is not None)

    def weyl_dimension(self, highest_weight: Sequence[int]) -> int:
        """Return the dimension of the irreducible module by Weyl's formula.

        ``highest_weight`` lists the nonnegative Dynkin labels.
        """
        if len(highest_weight) != self.rank or any(
            type(label) is not int or label < 0 for label in highest_weight
        ):
            raise EasyError(
                "weight", "one nonnegative integer per simple root is required"
            )
        lengths = self._symmetrizer()
        numerator = denominator = Fraction(1)
        for root in self.positive:
            numerator *= sum(
                (highest_weight[i] + 1) * root[i] * lengths[i] for i in range(self.rank)
            )
            denominator *= sum(root[i] * lengths[i] for i in range(self.rank))
        value = numerator / denominator
        assert value.denominator == 1
        return int(value)

    def __repr__(self) -> str:
        return f"RootSystem({self.family}{self.rank}, roots={self.count})"


def root_system(family: str, rank: int) -> RootSystem:
    """Return the root system of a simple Cartan type."""
    return RootSystem(family, rank)
