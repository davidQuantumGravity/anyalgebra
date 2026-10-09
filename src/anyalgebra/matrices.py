"""Matrices whose entries lie in an algebra.

The entries may come from any handle of :mod:`anyalgebra.easy`, including a
noncommutative or nonassociative one, so the order of every entry product is
fixed: ``(X @ Y)[i][j]`` is the sum over ``k`` of ``X[i][k] * Y[k][j]`` with
the left factor first.  Nothing here assumes that such a product is
associative::

    import anyalgebra.composition as ca
    import anyalgebra.matrices as am

    H = ca.quaternions()
    X = am.matrix(H, [[1, H.e1], [-H.e1, 2]])
    print(X.is_hermitian(), X.trace())  # True 3
    print(f"{X @ X:v}")  # the same matrix as coordinates
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction

from anyalgebra.easy import Algebra, EasyError, Element, Scalar

Entry = Element | int | Fraction


class Mat:
    """An exact rectangular matrix over one algebra handle."""

    __slots__ = ("algebra", "columns", "entries", "rows")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, algebra: Algebra, entries: Sequence[Sequence[Entry]]) -> None:
        rows = tuple(tuple(_entry(algebra, value) for value in row) for row in entries)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise EasyError("shape", "a matrix needs equally long, nonempty rows")
        self.algebra = algebra
        self.entries = rows
        self.rows = len(rows)
        self.columns = len(rows[0])

    # structure

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``."""
        return (self.rows, self.columns)

    def __getitem__(self, position: tuple[int, int]) -> Element:
        row, column = position
        return self.entries[row][column]

    def __iter__(self) -> Iterator[tuple[Element, ...]]:
        return iter(self.entries)

    def _same(self, other: object, *, shape: bool = True) -> Mat:
        if not isinstance(other, Mat):
            raise EasyError("parent", "the other operand is not a matrix")
        if other.algebra is not self.algebra:
            raise EasyError("parent", "the matrices have entries in different algebras")
        if shape and other.shape != self.shape:
            raise EasyError("shape", f"shapes {self.shape} and {other.shape} differ")
        return other

    def _map(self, function: object) -> Mat:
        apply = function  # a callable Element -> Element
        return Mat(self.algebra, [[apply(x) for x in row] for row in self.entries])  # type: ignore[operator]

    # arithmetic

    def __add__(self, other: object) -> Mat:
        right = self._same(other)
        return Mat(
            self.algebra,
            [
                [x + y for x, y in zip(row, other_row, strict=True)]
                for row, other_row in zip(self.entries, right.entries, strict=True)
            ],
        )

    def __neg__(self) -> Mat:
        return self._map(lambda x: -x)

    def __sub__(self, other: object) -> Mat:
        return self + (-self._same(other))

    def __mul__(self, scalar: object) -> Mat:
        if isinstance(scalar, Element):
            return self._map(lambda x: x * scalar)
        if type(scalar) is int or isinstance(scalar, Fraction):
            return self._map(lambda x: x * scalar)
        return NotImplemented

    def __rmul__(self, scalar: object) -> Mat:
        if isinstance(scalar, Element):
            return self._map(lambda x: scalar * x)
        if type(scalar) is int or isinstance(scalar, Fraction):
            return self._map(lambda x: x * scalar)
        return NotImplemented

    def __truediv__(self, scalar: Scalar) -> Mat:
        return self._map(lambda x: x / scalar)

    def __matmul__(self, other: object) -> Mat:
        right = self._same(other, shape=False)
        if self.columns != right.rows:
            raise EasyError(
                "shape", f"cannot multiply shapes {self.shape} and {right.shape}"
            )
        zero = self.algebra.zero
        return Mat(
            self.algebra,
            [
                [
                    sum(
                        (
                            self.entries[i][k] * right.entries[k][j]
                            for k in range(self.columns)
                        ),
                        zero,
                    )
                    for j in range(right.columns)
                ]
                for i in range(self.rows)
            ],
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Mat)
            and other.algebra is self.algebra
            and other.shape == self.shape
            and all(
                x == y
                for row, other_row in zip(self.entries, other.entries, strict=True)
                for x, y in zip(row, other_row, strict=True)
            )
        )

    def __bool__(self) -> bool:
        return any(bool(x) for row in self.entries for x in row)

    # transposes and conjugations

    def transpose(self) -> Mat:
        """Return the transpose, without conjugating the entries."""
        return Mat(self.algebra, list(zip(*self.entries, strict=True)))

    def conj(self, signs: Sequence[int] | None = None) -> Mat:
        """Conjugate every entry by the declared involution or by given signs."""
        if signs is None:
            return self._map(lambda x: x.conj())
        from anyalgebra.composition import conjugate

        return self._map(lambda x: conjugate(x, signs))

    def dagger(self, signs: Sequence[int] | None = None) -> Mat:
        """Return the conjugate transpose."""
        return self.conj(signs).transpose()

    def is_hermitian(self, signs: Sequence[int] | None = None) -> bool:
        """Decide whether the matrix equals its conjugate transpose."""
        return self.rows == self.columns and self == self.dagger(signs)

    def is_antihermitian(self, signs: Sequence[int] | None = None) -> bool:
        """Decide whether the matrix equals minus its conjugate transpose."""
        return self.rows == self.columns and self == -self.dagger(signs)

    def trace(self) -> Element:
        """Return the sum of the diagonal entries."""
        if self.rows != self.columns:
            raise EasyError("shape", "the trace needs a square matrix")
        return sum((self.entries[i][i] for i in range(self.rows)), self.algebra.zero)

    # printing

    def __format__(self, spec: str) -> str:
        from anyalgebra import easy

        stack = easy._DISPLAY
        name = spec or (
            stack[-1] if len(stack) > 1 else self.algebra.display or stack[0]
        )
        cells = [[format(x, name) for x in row] for row in self.entries]
        if name in ("latex", "l"):
            body = r" \\ ".join(" & ".join(row) for row in cells)
            return r"\begin{pmatrix} " + body + r" \end{pmatrix}"
        widths = [max(len(row[j]) for row in cells) for j in range(self.columns)]
        return "\n".join(
            "[" + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) + "]"
            for row in cells
        )

    def __str__(self) -> str:
        return format(self, "")

    def __repr__(self) -> str:
        return f"Mat({self.algebra.name}, {self.rows}x{self.columns})"

    def _repr_latex_(self) -> str:
        return f"${format(self, 'latex')}$"


def _entry(algebra: Algebra, value: object) -> Element:
    if isinstance(value, Element):
        if value.algebra is not algebra:
            raise EasyError("parent", "an entry belongs to a different algebra")
        return value
    if type(value) is int or isinstance(value, Fraction):
        return algebra.zero if value == 0 else algebra.scalar(value)
    raise EasyError("parent", f"{value!r} is not an entry of {algebra.name}")


# --- constructors ------------------------------------------------------------


def matrix(algebra: Algebra, entries: Sequence[Sequence[Entry]]) -> Mat:
    """Build a matrix from rows of elements or rational numbers."""
    return Mat(algebra, entries)


def zeros(algebra: Algebra, rows: int, columns: int | None = None) -> Mat:
    """Return the zero matrix."""
    width = rows if columns is None else columns
    return Mat(algebra, [[algebra.zero] * width for _ in range(rows)])


def identity(algebra: Algebra, size: int) -> Mat:
    """Return the identity matrix; the algebra must declare a unit."""
    return Mat(algebra, [[int(i == j) for j in range(size)] for i in range(size)])


def unit(algebra: Algebra, size: int, row: int, column: int, value: Entry = 1) -> Mat:
    """Return the square matrix with one entry ``value`` and zeros elsewhere."""
    if not (0 <= row < size and 0 <= column < size):
        raise EasyError("shape", "the position is outside the matrix")
    entry = _entry(algebra, value)
    return Mat(
        algebra,
        [
            [entry if (i, j) == (row, column) else algebra.zero for j in range(size)]
            for i in range(size)
        ],
    )


def hermitian_basis(algebra: Algebra, size: int) -> tuple[Mat, ...]:
    """Return a rational basis of the Hermitian ``size``-by-``size`` matrices.

    The basis lists the diagonal real units first, then for each position
    above the diagonal one matrix per basis element of the algebra.  Its
    length is ``n + n(n-1)/2 * dim A`` when the fixed points of the
    conjugation are the multiples of the unit.
    """
    return _symmetric_basis(algebra, size, sign=1)


def antihermitian_basis(algebra: Algebra, size: int) -> tuple[Mat, ...]:
    """Return a rational basis of the anti-Hermitian square matrices."""
    return _symmetric_basis(algebra, size, sign=-1)


def _symmetric_basis(algebra: Algebra, size: int, *, sign: int) -> tuple[Mat, ...]:
    signs = algebra.conjugation_signs
    if signs is None:
        raise EasyError("involution", f"{algebra.name} declares no conjugation")
    found = []
    for i in range(size):
        for element, own in zip(algebra.basis, signs, strict=True):
            if own == sign:
                found.append(unit(algebra, size, i, i, element))
    for i in range(size):
        for j in range(i + 1, size):
            for element in algebra.basis:
                upper = unit(algebra, size, i, j, element)
                found.append(upper + sign * unit(algebra, size, j, i, element.conj()))
    return tuple(found)


# --- named operations --------------------------------------------------------


def comm(x: Mat, y: Mat) -> Mat:
    """Return the matrix commutator ``X Y - Y X``."""
    return x @ y - y @ x


def anticomm(x: Mat, y: Mat) -> Mat:
    """Return the matrix anticommutator ``X Y + Y X``."""
    return x @ y + y @ x


def jordan(x: Mat, y: Mat) -> Mat:
    """Return the Jordan product ``(X Y + Y X) / 2``."""
    return anticomm(x, y) / 2


def inner(x: Mat, y: Mat) -> Fraction:
    """Return the real part of ``trace(X Y^dagger)``, a symmetric rational form."""
    value = (x @ y.dagger()).trace()
    index = x.algebra.unit_index
    if index is None:
        raise EasyError("unit", f"{x.algebra.name} declares no unit")
    return value.vector[index]


def coordinates(x: Mat) -> tuple[Fraction, ...]:
    """Return every entry's coordinates, row by row, as one rational vector."""
    return tuple(value for row in x.entries for entry in row for value in entry.vector)
