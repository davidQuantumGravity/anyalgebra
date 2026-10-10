"""Structure theory of a Lie algebra: Cartan subalgebras, roots, type, real form.

Given structure constants and nothing else, this module answers "which Lie
algebra is this?"::

    import anyalgebra.composition as ca
    import anyalgebra.lie_structure as ls
    import anyalgebra.matrix_lie as ml

    g = ml.sl(2, ca.octonions())
    print(ls.identify(g))  # D5, real form so(9,1)

Everything is exact.  Roots are found over the Gaussian rationals ``Q(i)``:
a compact direction of a real form has imaginary eigenvalues and a
noncompact one real eigenvalues.  The Cartan subalgebra used for the roots
is built from basis elements, and sums of two of them, whose adjoint maps
are diagonalizable with eigenvalues that are all rational or all rational
multiples of ``i``.  A basis in which no such elements span a Cartan
subalgebra is reported as an error, not guessed around;
:func:`cartan_subalgebra` works for every Lie algebra but does not give
roots.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from itertools import combinations, product
from math import isqrt

from anyalgebra.easy import EasyError
from anyalgebra.lie import LieAlgebra, Vector, _Span, cartan_matrix, inertia

Root = tuple[Fraction, ...]
Square = tuple[tuple[Fraction, ...], ...]


# --- Gaussian rationals ------------------------------------------------------


class Gauss:
    """A complex number with rational real and imaginary parts."""

    __slots__ = ("im", "re")

    def __init__(self, re: int | Fraction = 0, im: int | Fraction = 0) -> None:
        self.re, self.im = Fraction(re), Fraction(im)

    @staticmethod
    def _of(value: object) -> Gauss | None:
        if isinstance(value, Gauss):
            return value
        if type(value) is int or isinstance(value, Fraction):
            return Gauss(value)
        return None

    def __add__(self, other: object) -> Gauss:
        right = self._of(other)
        if right is None:
            return NotImplemented
        return Gauss(self.re + right.re, self.im + right.im)

    __radd__ = __add__

    def __neg__(self) -> Gauss:
        return Gauss(-self.re, -self.im)

    def __sub__(self, other: object) -> Gauss:
        right = self._of(other)
        if right is None:
            return NotImplemented
        return Gauss(self.re - right.re, self.im - right.im)

    def __rsub__(self, other: object) -> Gauss:
        left = self._of(other)
        if left is None:
            return NotImplemented
        return left - self

    def __mul__(self, other: object) -> Gauss:
        right = self._of(other)
        if right is None:
            return NotImplemented
        return Gauss(
            self.re * right.re - self.im * right.im,
            self.re * right.im + self.im * right.re,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Gauss:
        right = self._of(other)
        if right is None or not right:
            return NotImplemented
        norm = right.re * right.re + right.im * right.im
        return self * Gauss(right.re / norm, -right.im / norm)

    def conjugate(self) -> Gauss:
        """Return the complex conjugate."""
        return Gauss(self.re, -self.im)

    def __eq__(self, other: object) -> bool:
        right = self._of(other)
        if right is None:
            return NotImplemented
        return self.re == right.re and self.im == right.im

    def __hash__(self) -> int:
        return hash((self.re, self.im))

    def __bool__(self) -> bool:
        return bool(self.re or self.im)

    def __str__(self) -> str:
        if not self.im:
            return str(self.re)
        size = abs(self.im)
        imaginary = "i" if size == 1 else f"{size}*i"
        if not self.re:
            return ("-" if self.im < 0 else "") + imaginary
        return f"{self.re} {'-' if self.im < 0 else '+'} {imaginary}"

    def __repr__(self) -> str:
        return f"Gauss({self})"


GaussVector = tuple[Gauss, ...]


# --- rational linear algebra -------------------------------------------------


def _kernel(rows: Sequence[Sequence[Fraction]], width: int) -> list[Vector]:
    """Return a basis of the vectors ``x`` with ``rows @ x == 0``."""
    reduced: list[list[Fraction]] = []
    pivots: list[int] = []
    for source in rows:
        row = list(source)
        for pivot, other in zip(pivots, reduced, strict=True):
            factor = row[pivot]
            if factor:
                row = [a - factor * b for a, b in zip(row, other, strict=True)]
        pivot = next((index for index, value in enumerate(row) if value), -1)
        if pivot < 0:
            continue
        scale = row[pivot]
        row = [value / scale for value in row]
        for index, other in enumerate(reduced):
            factor = other[pivot]
            if factor:
                reduced[index] = [
                    a - factor * b for a, b in zip(other, row, strict=True)
                ]
        reduced.append(row)
        pivots.append(pivot)
    free = [column for column in range(width) if column not in pivots]
    basis = []
    for column in free:
        vector = [Fraction(0)] * width
        vector[column] = Fraction(1)
        for pivot, row in zip(pivots, reduced, strict=True):
            vector[pivot] = -row[column]
        basis.append(tuple(vector))
    return basis


def _times(a: Square, b: Square) -> Square:
    columns = list(zip(*b, strict=True))
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


def _combine(coefficients: Sequence[Fraction], vectors: Sequence[Vector]) -> Vector:
    total = [Fraction(0)] * len(vectors[0])
    for coefficient, vector in zip(coefficients, vectors, strict=True):
        if coefficient:
            for index, value in enumerate(vector):
                if value:
                    total[index] += coefficient * value
    return tuple(total)


def _restricted(algebra: LieAlgebra, x: Vector, space: Sequence[Vector]) -> Square:
    """Return the matrix of ``ad x`` on an invariant subspace, in its basis."""
    span = _Span(algebra.dimension)
    for vector in space:
        span.add(vector)
    columns = []
    for vector in space:
        found = span.coordinates(algebra.bracket(x, vector))
        if found is None:
            raise EasyError("closure", "the subspace is not invariant")
        columns.append(found)
    return tuple(zip(*columns, strict=True)) if columns else ()


def _divisors(value: int) -> list[int]:
    value = abs(value)
    found = []
    for candidate in range(1, isqrt(value) + 1):
        if value % candidate == 0:
            found += [candidate, value // candidate]
    return sorted(set(found))


def _rational_roots(coefficients: Sequence[Fraction]) -> list[Fraction]:
    """Return the rational roots of ``sum(c[k] * t^k)``."""
    scale = 1
    for value in coefficients:
        scale = scale * value.denominator // _gcd(scale, value.denominator)
    integers = [int(value * scale) for value in coefficients]
    roots: list[Fraction] = []
    if integers and integers[0] == 0:
        roots.append(Fraction(0))
        while integers and integers[0] == 0:
            integers = integers[1:]
    if len(integers) < 2:
        return roots
    if max(abs(integers[0]), abs(integers[-1])) > 10**12:
        raise EasyError("bound", "an eigenvalue polynomial has too large coefficients")
    for p in _divisors(integers[0]):
        for q in _divisors(integers[-1]):
            for candidate in (Fraction(p, q), Fraction(-p, q)):
                if candidate not in roots and not sum(
                    c * candidate**k for k, c in enumerate(integers)
                ):
                    roots.append(candidate)
    return roots


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _rational_sqrt(value: Fraction) -> Fraction | None:
    top, bottom = isqrt(value.numerator), isqrt(value.denominator)
    if top * top == value.numerator and bottom * bottom == value.denominator:
        return Fraction(top, bottom)
    return None


def _eigenspaces(matrix: Square) -> dict[Fraction, list[Vector]] | None:
    """Return the eigenspaces of a matrix that is diagonalizable over Q.

    ``None`` means that the rational eigenspaces do not fill the space.
    """
    size = len(matrix)
    if size == 0:
        return {}
    candidates: set[Fraction] = set()
    for attempt in range(3):
        vector = tuple(Fraction((k + 1) ** (attempt + 1)) for k in range(size))
        span = _Span(size)
        powers = []
        while span.add(vector):
            powers.append(vector)
            vector = tuple(
                sum(
                    (x * y for x, y in zip(row, vector, strict=True) if x and y),
                    Fraction(0),
                )
                for row in matrix
            )
        relation = span.coordinates(vector)
        assert relation is not None
        polynomial = [-value for value in relation] + [Fraction(1)]
        candidates |= set(_rational_roots(polynomial))
        found = {}
        total = 0
        for value in sorted(candidates):
            shifted = [
                [entry - (value if r == c else 0) for c, entry in enumerate(row)]
                for r, row in enumerate(matrix)
            ]
            kernel = _kernel(shifted, size)
            if kernel:
                found[value] = kernel
                total += len(kernel)
        if total == size:
            return found
    return None


# --- Cartan subalgebras of any Lie algebra -----------------------------------


def _fitting_null(
    algebra: LieAlgebra, x: Vector, space: Sequence[Vector]
) -> list[Vector]:
    """Return the Fitting null component of ``ad x`` on an invariant subspace."""
    matrix = _restricted(algebra, x, space)
    size = len(space)
    power, reached = matrix, 1
    while reached < size:
        power, reached = _times(power, power), reached * 2
    return [_combine(vector, space) for vector in _kernel(power, size)]


def _is_nilpotent_on(algebra: LieAlgebra, x: Vector, space: Sequence[Vector]) -> bool:
    return len(_fitting_null(algebra, x, space)) == len(space)


def _non_nilpotent(algebra: LieAlgebra, space: Sequence[Vector]) -> Vector | None:
    """Return an element of a subalgebra whose adjoint map there is not nilpotent.

    If every basis element and every sum of two is nilpotent, the subalgebra
    is nilpotent (de Graaf, "Lie Algebras: Theory and Algorithms", 2.4).
    """
    for vector in space:
        if not _is_nilpotent_on(algebra, vector, space):
            return vector
    for first, second in combinations(space, 2):
        total = tuple(a + b for a, b in zip(first, second, strict=True))
        if not _is_nilpotent_on(algebra, total, space):
            return total
    return None


def cartan_subalgebra(algebra: LieAlgebra) -> tuple[Vector, ...]:
    """Return a basis of a Cartan subalgebra of any Lie algebra.

    This is de Graaf's algorithm: descend through Fitting null components
    until the component is nilpotent.  The result is nilpotent and equal to
    its own normalizer.  Its dimension is the rank.
    """
    size = algebra.dimension
    whole = [tuple(Fraction(int(i == j)) for j in range(size)) for i in range(size)]
    x = _non_nilpotent(algebra, whole)
    if x is None:
        return tuple(whole)
    current = _fitting_null(algebra, x, whole)
    while True:
        y = _non_nilpotent(algebra, current)
        if y is None:
            return tuple(current)
        inside = _Span(size)
        for vector in current:
            inside.add(vector)
        for c in range(1, 4 * size + 8):
            candidate = tuple(a + c * (b - a) for a, b in zip(x, y, strict=True))
            smaller = _fitting_null(algebra, candidate, whole)
            if len(smaller) < len(current) and all(
                inside.coordinates(vector) is not None for vector in smaller
            ):
                x, current = candidate, smaller
                break
        else:  # pragma: no cover - only finitely many values of c can fail
            raise EasyError("bound", "no descending element was found")


def rank(algebra: LieAlgebra) -> int:
    """Return the rank: the dimension of a Cartan subalgebra."""
    return len(cartan_subalgebra(algebra))


def radical(algebra: LieAlgebra) -> tuple[Vector, ...]:
    """Return a basis of the solvable radical.

    The radical is the set of ``x`` with ``K(x, y) = 0`` for every ``y`` in
    ``[g, g]``, where ``K`` is the Killing form.
    """
    size = algebra.dimension
    form = algebra.killing_form()
    derived = _Span(size)
    kept = []
    for i, j in combinations(range(size), 2):
        vector = algebra.basis_bracket(i, j)
        if derived.add(vector):
            kept.append(vector)
    rows = [
        [
            sum((form[i][j] * y[j] for j in range(size) if y[j]), Fraction(0))
            for i in range(size)
        ]
        for y in kept
    ]
    return tuple(_kernel(rows, size))


# --- roots -------------------------------------------------------------------


class RootDecomposition:
    """A Cartan subalgebra that splits over ``Q(i)``, with roots and root vectors.

    ``torus`` is the basis ``h_1, ..., h_r`` of the Cartan subalgebra and
    ``kinds[k]`` says whether ``ad h_k`` has ``real`` or ``imaginary``
    eigenvalues.  A root is the tuple of rational numbers ``c_k`` with
    ``alpha(h_k) = c_k`` for a real direction and ``i * c_k`` for an
    imaginary one.  ``vectors[root]`` spans the root space.
    """

    def __init__(
        self,
        algebra: LieAlgebra,
        torus: Sequence[Vector],
        kinds: Sequence[str],
        vectors: dict[Root, GaussVector],
    ) -> None:
        self.algebra = algebra
        self.torus = tuple(torus)
        self.kinds = tuple(kinds)
        self.vectors = dict(vectors)
        self.roots: tuple[Root, ...] = tuple(sorted(vectors, reverse=True))
        self._set = set(self.roots)
        self.positive: tuple[Root, ...] = tuple(
            root for root in self.roots if next(v for v in root if v) > 0
        )
        positive = set(self.positive)
        sums = {
            tuple(a + b for a, b in zip(first, second, strict=True))
            for first, second in combinations(self.positive, 2)
        } | {tuple(2 * a for a in root) for root in self.positive}
        self.simple: tuple[Root, ...] = tuple(
            root for root in self.positive if root not in sums and root in positive
        )

    @property
    def rank(self) -> int:
        """Return the dimension of the Cartan subalgebra."""
        return len(self.torus)

    @property
    def is_split(self) -> bool:
        """Say whether every root is real, so the algebra is split over Q."""
        return all(
            kind == "real" or all(not root[k] for root in self.roots)
            for k, kind in enumerate(self.kinds)
        )

    def cartan_integer(self, beta: Root, alpha: Root) -> int:
        """Return ``<beta, alpha^vee>`` from the ``alpha``-string through ``beta``."""
        if beta == alpha:
            return 2
        if beta == tuple(-value for value in alpha):
            return -2

        def reach(sign: int) -> int:
            steps = 0
            while (
                tuple(
                    b + sign * (steps + 1) * a for a, b in zip(alpha, beta, strict=True)
                )
                in self._set
            ):
                steps += 1
            return steps

        return reach(-1) - reach(1)

    def cartan_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Return ``a_ij = <alpha_i, alpha_j^vee>`` for the simple roots."""
        return tuple(
            tuple(self.cartan_integer(first, second) for second in self.simple)
            for first in self.simple
        )

    def components(self) -> tuple[tuple[str, int], ...]:
        """Return the simple types of the complexification, as ``(family, rank)``."""
        return classify(self.cartan_matrix())

    @property
    def type(self) -> str:
        """Return the Cartan type of the complexification, such as ``A1+A1``."""
        parts = [f"{family}{size}" for family, size in self.components()]
        central = self.rank - len(self.simple)
        if central:
            parts.append(f"T{central}")
        return "+".join(parts) if parts else "0"

    def summands(self) -> tuple[str, ...]:
        """Return the simple real ideals by name, such as ``("su(2)", "su(2)")``.

        Complex conjugation permutes the simple components of the
        complexification.  A component it preserves is a real form, named
        from the Killing signature of its ideal.  Two components it swaps
        form one complex simple algebra regarded as real, named like
        ``sl(2,C)``.
        """
        matrix = self.cartan_matrix()
        parts = _components(matrix)
        form = self.algebra.killing_form()
        size = self.algebra.dimension

        def home(root: Root) -> int:
            return next(
                index
                for index, (_, nodes) in enumerate(parts)
                if any(self.cartan_integer(root, self.simple[n]) for n in nodes)
            )

        def conjugate(root: Root) -> Root:
            return tuple(
                -value if kind == "imaginary" else value
                for value, kind in zip(root, self.kinds, strict=True)
            )

        members: dict[int, list[Root]] = {index: [] for index in range(len(parts))}
        for root in self.roots:
            members[home(root)].append(root)
        names = []
        done: set[int] = set()
        for index, ((family, rank_), _) in enumerate(parts):
            if index in done:
                continue
            partner = home(conjugate(members[index][0]))
            done |= {index, partner}
            if partner != index:
                names.append(_complex_name(family, rank_))
                continue
            span = _Span(size)
            basis = []
            for root in members[index]:
                e = self.vectors[root]
                h = bracket(self.algebra, e, self.vectors[tuple(-v for v in root)])
                for vector in (e, h):
                    for part in (
                        tuple(value.re for value in vector),
                        tuple(value.im for value in vector),
                    ):
                        if span.add(part):
                            basis.append(part)
            images = [
                [
                    sum((form[i][j] * v[j] for j in range(size) if v[j]), Fraction(0))
                    for i in range(size)
                ]
                for v in basis
            ]
            restricted = [
                [
                    sum((u[i] * image[i] for i in range(size) if u[i]), Fraction(0))
                    for image in images
                ]
                for u in basis
            ]
            positive, negative, _ = inertia(restricted)
            found = real_forms(family, rank_, positive - negative)
            names.append(" or ".join(found) if found else f"{family}{rank_}")
        return tuple(names)

    def chevalley_generators(
        self,
    ) -> tuple[tuple[GaussVector, GaussVector, GaussVector], ...]:
        """Return ``(e_i, f_i, h_i)`` for each simple root.

        They satisfy ``[e_i, f_i] = h_i``, ``[h_i, e_j] = a_ji e_j`` and
        ``[h_i, f_j] = -a_ji f_j`` with the matrix of :meth:`cartan_matrix`.
        Coordinates are Gaussian rationals; they are rational when the
        decomposition is split.
        """
        found = []
        for root in self.simple:
            e = self.vectors[root]
            f = self.vectors[tuple(-value for value in root)]
            h = bracket(self.algebra, e, f)
            image = bracket(self.algebra, h, e)
            index = next(k for k, value in enumerate(e) if value)
            factor = Gauss(2) / (image[index] / e[index])
            found.append(
                (e, tuple(value * factor for value in f), tuple(v * factor for v in h))
            )
        return tuple(found)

    def __repr__(self) -> str:
        return (
            f"RootDecomposition({self.algebra.name}, type={self.type}, "
            f"roots={len(self.roots)})"
        )


def bracket(algebra: LieAlgebra, u: Sequence[Gauss], v: Sequence[Gauss]) -> GaussVector:
    """Return the bracket of two vectors with Gaussian rational coordinates."""
    total = [Gauss() for _ in range(algebra.dimension)]
    for i, a in enumerate(u):
        if a:
            for j, b in enumerate(v):
                if b:
                    scale = a * b
                    for k, c in enumerate(algebra.basis_bracket(i, j)):
                        if c:
                            total[k] = total[k] + scale * c
    return tuple(total)


def _split_step(
    algebra: LieAlgebra, x: Vector, spaces: dict[Root, list[Vector]]
) -> tuple[str, dict[Root, list[Vector]]] | None:
    """Refine joint eigenspaces by ``(ad x)^2``; ``None`` if ``x`` does not qualify."""
    refined: dict[Root, list[Vector]] = {}
    signs = set()
    for key, space in spaces.items():
        matrix = _restricted(algebra, x, space)
        found = _eigenspaces(_times(matrix, matrix))
        if found is None:
            return None
        for value, kernel in found.items():
            if value == 0:
                # ad x must vanish, not merely square to zero.
                if any(
                    any(
                        sum(
                            (m * v for m, v in zip(row, vector, strict=True)),
                            Fraction(0),
                        )
                        for row in matrix
                    )
                    for vector in kernel
                ):
                    return None
            else:
                if _rational_sqrt(abs(value)) is None:
                    return None
                signs.add(value > 0)
            refined[(*key, value)] = [_combine(vector, space) for vector in kernel]
    if len(signs) > 1:
        return None
    return ("imaginary" if signs == {False} else "real"), refined


def root_decomposition(algebra: LieAlgebra) -> RootDecomposition:
    """Return a Cartan subalgebra with its roots and root vectors.

    The algebra must be reductive.  See the module description for the
    elements the search considers and the error it raises otherwise.
    """
    size = algebra.dimension
    spaces: dict[Root, list[Vector]] = {
        (): [tuple(Fraction(int(i == j)) for j in range(size)) for i in range(size)]
    }
    torus: list[Vector] = []
    kinds: list[str] = []
    chosen = _Span(size)
    while True:
        zero = (Fraction(0),) * len(torus)
        centralizer = spaces[zero]
        if len(centralizer) == len(torus):
            break
        candidates: Iterable[Vector] = (
            *centralizer,
            *(
                tuple(a + b for a, b in zip(first, second, strict=True))
                for first, second in combinations(centralizer, 2)
            ),
            *(
                tuple(a - b for a, b in zip(first, second, strict=True))
                for first, second in combinations(centralizer, 2)
            ),
        )
        for candidate in candidates:
            if chosen.coordinates(candidate) is not None:
                continue
            step = _split_step(algebra, candidate, spaces)
            if step is not None:
                chosen.add(candidate)
                torus.append(candidate)
                kinds.append(step[0])
                spaces = step[1]
                break
        else:
            raise EasyError(
                "split",
                f"no Cartan subalgebra of {algebra.name} that splits over Q(i) was "
                "found among the basis elements and sums of two of them",
            )
    vectors: dict[Root, GaussVector] = {}
    units = [Gauss(1) if kind == "real" else Gauss(0, 1) for kind in kinds]
    for key, space in spaces.items():
        if not any(key):
            continue
        lengths = [_rational_sqrt(abs(value)) for value in key]
        matrices = [_restricted(algebra, x, space) for x in torus]
        count = 0
        active = [k for k, value in enumerate(key) if value]
        for pattern in product((1, -1), repeat=len(active)):
            images = [
                tuple(Gauss(int(i == j)) for j in range(len(space)))
                for i in range(len(space))
            ]
            for sign, k in zip(pattern, active, strict=True):
                length = lengths[k]
                assert length is not None
                eigenvalue = units[k] * (sign * length)
                images = [
                    tuple(
                        (
                            sum(
                                (
                                    Gauss(m) * v
                                    for m, v in zip(row, vector, strict=True)
                                    if m
                                ),
                                Gauss(),
                            )
                            / eigenvalue
                            + vector[r]
                        )
                        for r, row in enumerate(matrices[k])
                    )
                    for vector in images
                ]
                images = [vector for vector in images if any(vector)]
            if not images:
                continue
            count += 1
            root = [Fraction(0)] * len(key)
            for sign, k in zip(pattern, active, strict=True):
                length = lengths[k]
                assert length is not None
                root[k] = sign * length
            local = images[0]
            vectors[tuple(root)] = tuple(
                sum((local[j] * Gauss(space[j][k]) for j in range(len(space))), Gauss())
                for k in range(size)
            )
        if count != len(space):
            raise EasyError(
                "reductive", f"{algebra.name} has a root space of dimension above one"
            )
    return RootDecomposition(algebra, torus, kinds, vectors)


# --- classification ----------------------------------------------------------


def classify(matrix: Sequence[Sequence[int]]) -> tuple[tuple[str, int], ...]:
    """Return the simple types of a Cartan matrix, largest component first.

    Each component is named from its Dynkin diagram.  ``B2`` is used for the
    type that is also ``C2``, and ``A1``, ``A3`` for ``B1`` and ``D3``.
    """
    found = [kind for kind, _ in _components(matrix)]
    return tuple(sorted(found, key=lambda item: (-item[1], item[0])))


def _components(
    matrix: Sequence[Sequence[int]],
) -> list[tuple[tuple[str, int], list[int]]]:
    size = len(matrix)
    seen: set[int] = set()
    found = []
    for start in range(size):
        if start in seen:
            continue
        component = [start]
        for node in component:
            component += [
                other
                for other in range(size)
                if other != node and matrix[node][other] and other not in component
            ]
        seen |= set(component)
        found.append((_component_type(matrix, component), component))
    return found


def _complex_name(family: str, size: int) -> str:
    classical = {
        "A": f"sl({size + 1},C)",
        "B": f"so({2 * size + 1},C)",
        "C": f"sp({2 * size},C)",
        "D": f"so({2 * size},C)",
    }
    return classical.get(family, f"{family.lower()}{size}(C)")


def _component_type(
    matrix: Sequence[Sequence[int]], nodes: Sequence[int]
) -> tuple[str, int]:
    size = len(nodes)
    neighbours = {i: [j for j in nodes if j != i and matrix[i][j]] for i in nodes}
    bonds = {(i, j): matrix[i][j] * matrix[j][i] for i in nodes for j in neighbours[i]}
    strongest = max(bonds.values(), default=1)
    if strongest == 3:
        return ("G", 2)
    if strongest == 2:
        i, j = next(pair for pair, value in bonds.items() if value == 2)
        if size == 2:
            return ("B", 2)
        if len(neighbours[i]) == 2 and len(neighbours[j]) == 2:
            return ("F", 4)
        end, inner = (i, j) if len(neighbours[i]) == 1 else (j, i)
        # |a_ij| = 2 says that root i is the long one of the pair.
        return ("C" if abs(matrix[end][inner]) == 2 else "B", size)
    branch = [i for i in nodes if len(neighbours[i]) == 3]
    if not branch:
        return ("A", size)
    arms = []
    for first in neighbours[branch[0]]:
        length, previous, node = 1, branch[0], first
        while len(neighbours[node]) == 2:
            previous, node = node, next(n for n in neighbours[node] if n != previous)
            length += 1
        arms.append(length)
    arms.sort()
    if arms[:2] == [1, 1]:
        return ("D", size)
    return ("E", size)


def _real_forms(family: str, size: int) -> dict[int, list[str]]:
    """Return the real forms of a simple type, keyed by character.

    The character is the number of noncompact minus the number of compact
    directions.  Two forms can share a character, such as ``so(12,6)`` and
    ``so*(18)``; then both names are listed.
    """
    forms: dict[int, list[str]] = {}

    def add(character: int, name: str) -> None:
        forms.setdefault(character, []).append(name)

    def orthogonal(total: int) -> None:
        for q in range(total // 2 + 1):
            p = total - q
            character = p * q - (p * (p - 1) + q * (q - 1)) // 2
            add(character, f"so({p})" if q == 0 else f"so({p},{q})")

    if family == "A":
        n = size + 1
        for q in range(n // 2 + 1):
            p = n - q
            if (p, q) != (1, 1):  # su(1,1) is sl(2,R), listed below
                add(1 - (p - q) ** 2, f"su({p})" if q == 0 else f"su({p},{q})")
        add(size, f"sl({n},R)")
        if n % 2 == 0 and n > 2:
            add(-n - 1, f"su*({n})")
    elif family == "B":
        orthogonal(2 * size + 1)
    elif family == "C":
        add(size, f"sp({2 * size},R)")
        for q in range(size // 2 + 1):
            p = size - q
            character = 4 * p * q - p * (2 * p + 1) - q * (2 * q + 1)
            add(character, f"sp({p})" if q == 0 else f"sp({p},{q})")
    elif family == "D":
        orthogonal(2 * size)
        add(-size, f"so*({2 * size})")
    else:
        table = {
            ("G", 2): (-14, 2),
            ("F", 4): (-52, -20, 4),
            ("E", 6): (-78, -26, -14, 2, 6),
            ("E", 7): (-133, -25, -5, 7),
            ("E", 8): (-248, -24, 8),
        }
        name = f"{family.lower()}{size}"
        for character in table[(family, size)]:
            compact = character == min(table[(family, size)])
            add(character, name if compact else f"{name}({character})")
    return forms


def real_forms(family: str, size: int, character: int) -> tuple[str, ...]:
    """Return the names of the real forms of a simple type with a character."""
    if (family, size) not in {("G", 2), ("F", 4), ("E", 6), ("E", 7), ("E", 8)}:
        cartan_matrix(family, size)
    return tuple(_real_forms(family, size).get(character, ()))


def identify(algebra: LieAlgebra) -> str:
    """Name a semisimple real Lie algebra from its structure constants.

    The answer gives the Cartan type of the complexification and the real
    form of each simple ideal, read off the Killing signature, as in
    ``E6, real form e6(-26)`` or ``A1+A1, real form sl(2,C)``.
    """
    if algebra.killing_signature()[2]:
        raise EasyError(
            "semisimple", f"{algebra.name} is not semisimple; see radical()"
        )
    decomposition = root_decomposition(algebra)
    return f"{decomposition.type}, real form " + " + ".join(decomposition.summands())
