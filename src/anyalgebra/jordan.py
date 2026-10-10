"""Hermitian Jordan algebras and tensor-Jordan carriers.

``hermitian(A, n)`` is the algebra ``J_n(A)`` of Hermitian ``n``-by-``n``
matrices over an algebra with involution, under the Jordan product
``X o Y = (XY + YX) / 2``::

    import anyalgebra.composition as ca
    import anyalgebra.jordan as aj

    J = aj.hermitian(ca.octonions(), 3)  # the 27-dimensional Albert algebra
    print(J.rank, aj.is_jordan(J))  # 27 True

The identities are decided exactly, never sampled: each law is replaced by
its full linearization, which the basis then settles over the rationals.

``tensor_jordan(A, n, B)`` is the carrier ``A tensor J_n(B)`` with the
factor-wise product.  It is in general not a Jordan algebra, and the same
exact tests say when it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations_with_replacement, product

from anyalgebra.easy import Algebra, EasyError, Element, algebra

Sparse = dict[int, Fraction]


class _Table:
    """A fast sparse copy of a handle's multiplication table."""

    def __init__(self, source: Algebra) -> None:
        self.rank = source.rank
        self.cells: list[list[Sparse]] = [
            [_sparse((a * b).vector) for b in source.basis] for a in source.basis
        ]

    def multiply(self, u: Sparse, v: Sparse) -> Sparse:
        """Return the product of two sparse coordinate vectors."""
        return _combine(
            *((a * b, self.cells[i][j]) for i, a in u.items() for j, b in v.items())
        )


def _sparse(vector: Sequence[Fraction]) -> Sparse:
    return {index: value for index, value in enumerate(vector) if value}


def _combine(*terms: tuple[int | Fraction, Sparse]) -> Sparse:
    total: Sparse = {}
    for scale, vector in terms:
        for k, value in vector.items():
            updated = total.get(k, 0) + scale * value
            if updated:
                total[k] = updated
            else:
                total.pop(k, None)
    return total


# --- Hermitian matrices ------------------------------------------------------


def hermitian(source: Algebra, size: int, *, name: str | None = None) -> Algebra:
    """Return ``J_n(A)``: Hermitian matrices over ``source``, Jordan product.

    The basis lists the diagonal units ``d1 .. dn`` and then, for each
    position ``i < j`` and each basis element ``a`` of the algebra, the matrix
    with ``a`` at ``(i, j)`` and its conjugate at ``(j, i)``, labeled ``xij``
    for the unit and ``xij.a`` otherwise.  The algebra's unit is
    ``d1 + ... + dn``; see :func:`one`.
    """
    signs = source.conjugation_signs
    if signs is None or source.unit_index is None:
        raise EasyError("involution", "a declared conjugation and unit are required")
    if size < 1 or any(
        sign != (1 if k == source.unit_index else -1) for k, sign in enumerate(signs)
    ):
        raise EasyError(
            "involution",
            "the conjugation must fix exactly the multiples of the unit, and n >= 1",
        )
    entries = _Table(source)
    unit = source.unit_index
    positions = [(i, j) for i in range(size) for j in range(i + 1, size)]
    labels = [f"d{i + 1}" for i in range(size)]
    for i, j in positions:
        for k, label in enumerate(source.labels):
            labels.append(
                f"x{i + 1}{j + 1}" if k == unit else f"x{i + 1}{j + 1}.{label}"
            )
    width = source.rank

    def as_matrix(index: int) -> dict[tuple[int, int], Sparse]:
        if index < size:
            return {(index, index): {unit: Fraction(1)}}
        (i, j), k = positions[(index - size) // width], (index - size) % width
        return {(i, j): {k: Fraction(1)}, (j, i): {k: Fraction(signs[k])}}

    def matmul(
        x: dict[tuple[int, int], Sparse], y: dict[tuple[int, int], Sparse]
    ) -> dict[tuple[int, int], Sparse]:
        total: dict[tuple[int, int], Sparse] = {}
        for (i, k), a in x.items():
            for (m, j), b in y.items():
                if k == m:
                    total[(i, j)] = _combine(
                        (1, total.get((i, j), {})), (1, entries.multiply(a, b))
                    )
        return total

    def coordinates(x: dict[tuple[int, int], Sparse]) -> dict[str, Fraction]:
        found: dict[str, Fraction] = {}
        for i in range(size):
            value = x.get((i, i), {}).get(unit, 0)
            if value:
                found[labels[i]] = Fraction(value)
        for number, (i, j) in enumerate(positions):
            for k, value in x.get((i, j), {}).items():
                found[labels[size + number * width + k]] = value
        return found

    matrices = [as_matrix(index) for index in range(len(labels))]
    table = {}
    for a, x in enumerate(matrices):
        for b, y in enumerate(matrices):
            xy, yx = matmul(x, y), matmul(y, x)
            half = {
                key: _combine(
                    (Fraction(1, 2), xy.get(key, {})), (Fraction(1, 2), yx.get(key, {}))
                )
                for key in set(xy) | set(yx)
            }
            cell = coordinates(half)
            if cell:
                table[(labels[a], labels[b])] = cell
    return algebra(labels, table, name=name or f"J{size}({source.name})", unit=None)


def diagonal_count(source: Algebra) -> int:
    """Return ``n`` for an algebra built by :func:`hermitian`."""
    count = sum(1 for label in source.labels if label[0] == "d" and label[1:].isdigit())
    if count == 0 or source.labels[:count] != tuple(f"d{i + 1}" for i in range(count)):
        raise EasyError("structure", f"{source.name} was not built by hermitian()")
    return count


def one(source: Algebra) -> Element:
    """Return the unit ``d1 + ... + dn`` of a Hermitian Jordan algebra."""
    return sum(source.basis[1 : diagonal_count(source)], source.basis[0])


def trace(x: Element) -> Fraction:
    """Return the matrix trace, the sum of the diagonal coordinates."""
    return sum(x.vector[: diagonal_count(x.algebra)], Fraction(0))


def trace_form(x: Element, y: Element) -> Fraction:
    """Return the symmetric bilinear form ``trace(x o y)``."""
    return trace(x * y)


def freudenthal(x: Element, y: Element) -> Element:
    """Return the Freudenthal cross product of two elements of ``J_3(A)``.

    ``x # y = x o y - (tr(x) y + tr(y) x)/2 + (tr(x) tr(y) - tr(x o y))/2 * 1``
    """
    if diagonal_count(x.algebra) != 3:
        raise EasyError(
            "structure", "the Freudenthal product is defined on 3-by-3 matrices"
        )
    tx, ty = trace(x), trace(y)
    return (
        x * y
        - (tx * y + ty * x) / 2
        + one(x.algebra) * ((tx * ty - trace_form(x, y)) / 2)
    )


def determinant(x: Element) -> Fraction:
    """Return the cubic norm ``trace(x o (x # x)) / 3`` of an element of ``J_3(A)``."""
    return trace_form(x, freudenthal(x, x)) / 3


# --- exact law tests ---------------------------------------------------------


def is_commutative(source: Algebra) -> bool:
    """Decide exactly whether the product is commutative."""
    table = _Table(source)
    return all(
        table.cells[i][j] == table.cells[j][i]
        for i in range(source.rank)
        for j in range(i + 1, source.rank)
    )


def satisfies_jordan_identity(source: Algebra) -> bool:
    """Decide exactly whether ``((x o x) o y) o x = (x o x) o (y o x)`` always holds.

    The identity is cubic in ``x``.  Its full linearization says that for
    every unordered basis triple ``a, b, c`` the operator
    ``[R_c, L_ab] + [R_a, L_bc] + [R_b, L_ca]`` vanishes, where ``L_uv`` is
    left multiplication by ``u o v + v o u`` and ``R_c`` right multiplication
    by ``c``.  The commutators ``[R_c, L_k]`` are computed once per basis pair.
    """
    table = _Table(source)
    rank = source.rank
    # left[k][j] = e_k * e_j and right[c][j] = e_j * e_c, as sparse columns
    left = [[table.cells[k][j] for j in range(rank)] for k in range(rank)]
    right = [[table.cells[j][c] for j in range(rank)] for c in range(rank)]

    def apply(columns: list[Sparse], vector: Sparse) -> Sparse:
        return _combine(*((value, columns[j]) for j, value in vector.items()))

    # commutator[k][c][j] = R_c(L_k e_j) - L_k(R_c e_j)
    commutator = [
        [
            [
                _combine(
                    (1, apply(right[c], left[k][j])), (-1, apply(left[k], right[c][j]))
                )
                for j in range(rank)
            ]
            for c in range(rank)
        ]
        for k in range(rank)
    ]
    squares = [
        [_combine((1, table.cells[u][v]), (1, table.cells[v][u])) for v in range(rank)]
        for u in range(rank)
    ]
    for a, b, c in combinations_with_replacement(range(rank), 3):
        terms = [
            (value, commutator[k][lone])
            for pair, lone in (((a, b), c), ((b, c), a), ((c, a), b))
            for k, value in squares[pair[0]][pair[1]].items()
        ]
        for j in range(rank):
            if _combine(*((value, columns[j]) for value, columns in terms)):
                return False
    return True


def is_jordan(source: Algebra) -> bool:
    """Decide exactly whether the algebra is commutative and Jordan."""
    return is_commutative(source) and satisfies_jordan_identity(source)


# --- tensor-Jordan carriers --------------------------------------------------


def tensor_jordan(
    left: Algebra, size: int, right: Algebra, *, name: str | None = None
) -> Algebra:
    """Return the carrier ``left tensor J_n(right)`` with the factor-wise product.

    The basis runs with the Jordan factor fastest and is labeled ``a|x``.
    The dimension is ``dim(left) * (n + n(n-1)/2 * dim(right))``.
    """
    jordan_factor = hermitian(right, size)
    first, second = _Table(left), _Table(jordan_factor)
    width = jordan_factor.rank
    labels = [f"{a}|{x}" for a, x in product(left.labels, jordan_factor.labels)]
    table = {}
    for (i, j), (k, m) in product(product(range(left.rank), range(width)), repeat=2):
        cell = {
            labels[a * width + b]: x * y
            for a, x in first.cells[i][k].items()
            for b, y in second.cells[j][m].items()
        }
        if cell:
            table[(labels[i * width + j], labels[k * width + m])] = cell
    default = f"{left.name}xJ{size}({right.name})"
    return algebra(labels, table, name=name or default, unit=None)


def tensor_jordan_dimension(
    left_dimension: int, size: int, right_dimension: int
) -> int:
    """Return ``dim(A) * (n + n(n-1)/2 * dim(B))``."""
    return left_dimension * (size + size * (size - 1) // 2 * right_dimension)


# --- reports -----------------------------------------------------------------


def report(source: Algebra) -> str:
    """Return the exact Jordan facts of a commutative algebra as text."""
    commutative = is_commutative(source)
    rows = (
        ("dimension", source.rank),
        ("commutative", commutative),
        ("Jordan identity", satisfies_jordan_identity(source)),
        ("Jordan algebra", commutative and satisfies_jordan_identity(source)),
    )
    width = max(len(title) for title, _ in rows)
    return "\n".join(
        [
            f"{source.name}:",
            *(f"  {title.ljust(width)}  {value}" for title, value in rows),
        ]
    )


def proof_table(algebras: Sequence[Algebra], sizes: Sequence[int] = (2, 3)) -> str:
    """Return a table saying for which ``A`` and ``n`` the algebra ``J_n(A)`` is Jordan.

    Every cell is decided exactly by :func:`is_jordan`.
    """
    heads = [f"J{size}" for size in sizes]
    cells = [
        [
            "Jordan" if is_jordan(hermitian(source, size)) else "not Jordan"
            for size in sizes
        ]
        for source in algebras
    ]
    names = [source.name for source in algebras]
    side = max(len(name) for name in names) + 2
    width = max(len(text) for text in [*heads, *(c for row in cells for c in row)]) + 2
    lines = ["".ljust(side) + "".join(head.ljust(width) for head in heads)]
    for name, row in zip(names, cells, strict=True):
        lines.append(name.ljust(side) + "".join(cell.ljust(width) for cell in row))
    return "\n".join(line.rstrip() for line in lines)
