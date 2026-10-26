"""Subalgebras, branching rules and chains for complex Lie algebras.

An :class:`Embedding` of ``h`` in ``g`` is a projection matrix that sends
weights of ``g`` to weights of ``h``.  Branching a representation projects
its weights and decomposes the result::

    import anyalgebra.branching as ab
    import anyalgebra.weights as aw

    E6 = aw.lie_type("E6")
    step = ab.regular(E6, 0, extended=False)  # D5+U1
    print(step.target.name, "|", step.show((1, 0, 0, 0, 0, 0)))

Projection matrices are *derived*, never typed in:

* regular subalgebras, from the Dynkin diagram or the extended diagram
  (Borel and de Siebenthal);
* a subalgebra of ``sl``, ``so`` or ``sp`` given by a representation;
* the principal ``A1`` and the two foldings ``E6 > F4`` and ``D4 > G2``;
* any subalgebra given by explicit elements of a Lie algebra, through
  :mod:`anyalgebra.lie_structure`.

Nodes of a diagram count from zero in the Bourbaki numbering; the extra node
of an extended diagram has the number ``rank``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from itertools import permutations
from math import gcd, lcm

import anyalgebra.lie_structure as ls
from anyalgebra.easy import EasyError
from anyalgebra.lie import LieAlgebra, _Span, cartan_matrix, inverse
from anyalgebra.weights import Character, LieType, Number, Weight, lie_type

Row = tuple[Fraction, ...]

PRIMES = (2, 3, 5)


# --- diagrams ----------------------------------------------------------------


def canonical_order(
    matrix: Sequence[Sequence[int]],
) -> list[tuple[tuple[str, int], list[int]]]:
    """Return the simple types of a Cartan matrix with their nodes renumbered.

    Each component comes with its nodes listed in the Bourbaki order of its
    type, so that the submatrix in that order is the standard Cartan matrix.
    """
    found = []
    for (family, size), nodes in ls._components(matrix):
        standard = cartan_matrix(family, size)
        for order in permutations(nodes):
            if all(
                matrix[order[i]][order[j]] == standard[i][j]
                for i in range(size)
                for j in range(size)
            ):
                found.append(((family, size), list(order)))
                break
        else:  # pragma: no cover - classify and cartan_matrix agree
            raise EasyError("type", "the diagram does not match its type")
    return found


class Embedding:
    """A subalgebra ``target`` of ``source``, as a projection of weights."""

    def __init__(
        self,
        source: LieType,
        target: LieType,
        projection: Sequence[Sequence[Number]],
        label: str = "",
    ) -> None:
        rows = tuple(tuple(Fraction(v) for v in row) for row in projection)
        if len(rows) != target.rank or any(len(row) != source.rank for row in rows):
            raise EasyError("shape", "the projection does not fit the two types")
        self.source, self.target, self.projection = source, target, rows
        self.label = label or f"{source.name} > {target.name}"

    def project(self, weight: Sequence[Number]) -> Weight:
        """Return the weight of the subalgebra that a weight restricts to."""
        return tuple(
            sum((p * w for p, w in zip(row, weight, strict=True) if p), Fraction(0))
            for row in self.projection
        )

    def branch(self, highest: Sequence[Number]) -> Character:
        """Return the content of a representation restricted to the subalgebra."""
        projected: Character = {}
        for weight, multiplicity in self.source.character(highest).items():
            image = self.project(weight)
            projected[image] = projected.get(image, 0) + multiplicity
        return self.target.decompose(projected)

    def show(self, highest: Sequence[Number]) -> str:
        """Return a branching rule as text, such as ``27 = 16(1) + 10(-2) + 1(4)``."""
        return (
            f"{self.source.label(highest)} = {self.target.show(self.branch(highest))}"
        )

    def then(self, other: Embedding) -> Embedding:
        """Return the composite with a further embedding into the target."""
        if other.source != self.target:
            raise EasyError("shape", "the second embedding starts from another type")
        rows = [
            [
                sum(
                    (a * self.projection[k][j] for k, a in enumerate(row) if a),
                    Fraction(0),
                )
                for j in range(self.source.rank)
            ]
            for row in other.projection
        ]
        return Embedding(
            self.source, other.target, rows, f"{self.label} > {other.target.name}"
        )

    def index(self) -> tuple[Fraction, ...]:
        """Return the embedding index of each simple factor of the subalgebra.

        The Dynkin index of a restricted representation is the index of the
        representation times this number.  The source must be simple.
        """
        first = [0] * self.source.rank
        first[0] = 1
        content = self.branch(first)
        whole = self.source.index(first)
        found = []
        for offset, (family, size) in zip(
            self.target.offsets, self.target.factors, strict=True
        ):
            single = LieType([(family, size)])
            total = Fraction(0)
            for weight, count in content.items():
                piece = weight[offset : offset + size]
                others = self.target.dimension(weight) // single.dimension(piece)
                total += count * others * single.index(piece)
            found.append(total / whole)
        return tuple(found)

    def __repr__(self) -> str:
        return f"Embedding({self.label})"


def _assemble(
    source: LieType,
    pieces: Sequence[tuple[tuple[str, int], Sequence[Sequence[Number]]]],
    charges: Sequence[Sequence[Number]],
    label: str = "",
) -> Embedding:
    """Build an embedding from rows grouped by simple factor, then charge rows."""
    order = sorted(
        range(len(pieces)), key=lambda k: (-pieces[k][0][1], pieces[k][0][0])
    )
    factors = [pieces[k][0] for k in order]
    rows = [row for k in order for row in pieces[k][1]] + list(charges)
    return Embedding(source, LieType(factors, len(charges)), rows, label)


def _simple(source: LieType) -> None:
    if not source.is_simple:
        raise EasyError("type", "a simple type is required; see on_factor")


def marks(source: LieType) -> tuple[int, ...]:
    """Return the coefficients of the highest root in the simple roots."""
    _simple(source)
    return source.positive_roots[-1][0]


def _primitive(row: Sequence[Fraction]) -> list[Fraction]:
    scale = lcm(*(value.denominator for value in row))
    whole = [int(value * scale) for value in row]
    common = gcd(*whole) or 1
    return [Fraction(value, common) for value in whole]


def regular(source: LieType, node: int, *, extended: bool = True) -> Embedding:
    """Return the regular subalgebra obtained by deleting one node.

    With ``extended`` the node is deleted from the extended diagram and the
    result is semisimple of the same rank.  Without it the node is deleted
    from the ordinary diagram and replaced by a ``U1`` whose charge is the
    coefficient of that simple root, scaled to coprime integers.
    """
    _simple(source)
    size = source.rank
    last = size + 1 if extended else size
    if not 0 <= node < last:
        raise EasyError("shape", f"{source.name} has no node {node} to delete")
    theta = marks(source)
    lengths = source.lengths
    # Each kept node is a root: (coefficients, row of the projection).
    unit = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    nodes = [(tuple(int(i == j) for j in range(size)), unit[i]) for i in range(size)]
    if extended:
        nodes.append(
            (
                tuple(-m for m in theta),
                [-Fraction(theta[i]) * lengths[i] / 2 for i in range(size)],
            )
        )
    kept = [entry for index, entry in enumerate(nodes) if index != node]

    def form(a: Sequence[int], b: Sequence[int]) -> Fraction:
        return sum(
            (
                a[i] * b[j] * source.cartan[i][j] * lengths[j] / 2
                for i in range(size)
                for j in range(size)
                if a[i] and b[j]
            ),
            Fraction(0),
        )

    matrix = [[int(2 * form(a, b) / form(b, b)) for b, _ in kept] for a, _ in kept]
    pieces = [
        (kind, [kept[index][1] for index in order])
        for kind, order in canonical_order(matrix)
    ]
    charges = []
    if not extended:
        charges.append(
            _primitive([source.inverse_cartan[j][node] for j in range(size)])
        )
    where = f"node {node}" + (" of the extended diagram" if extended else "")
    found = _assemble(source, pieces, charges)
    found.label = f"{source.name} > {found.target.name} ({where})"
    return found


def on_factor(source: LieType, position: int, inner: Embedding) -> Embedding:
    """Apply an embedding to one simple factor of a larger type."""
    family, size = source.factors[position]
    if inner.source != LieType([(family, size)]):
        raise EasyError("shape", "the embedding does not start from that factor")
    offset = source.offsets[position]

    def widen(row: Sequence[Fraction], start: int) -> list[Fraction]:
        full = [Fraction(0)] * source.rank
        full[start : start + len(row)] = row
        return full

    pieces = []
    for k, (factor, start) in enumerate(
        zip(source.factors, source.offsets, strict=True)
    ):
        if k != position:
            identity = [
                [Fraction(int(i == j)) for j in range(factor[1])]
                for i in range(factor[1])
            ]
            pieces.append((factor, [widen(row, start) for row in identity]))
    inner_rank = inner.target.semisimple_rank
    for start, factor in zip(inner.target.offsets, inner.target.factors, strict=True):
        rows = inner.projection[start : start + factor[1]]
        pieces.append((factor, [widen(row, offset) for row in rows]))
    charges = [widen(row, offset) for row in inner.projection[inner_rank:]]
    for k in range(source.abelian):
        full = [Fraction(0)] * source.rank
        full[source.semisimple_rank + k] = Fraction(1)
        charges.append(full)
    return _assemble(source, pieces, charges)


# --- special subalgebras -----------------------------------------------------


def principal_sl2(source: LieType) -> Embedding:
    """Return the principal ``A1``: every simple root restricts to its root."""
    _simple(source)
    size = source.rank
    row = [
        2 * sum((source.inverse_cartan[i][j] for j in range(size)), Fraction(0))
        for i in range(size)
    ]
    return Embedding(source, lie_type("A1"), [row], f"{source.name} > A1 (principal)")


_FOLDINGS = {
    "E6": ("F4", ((1,), (3,), (2, 4), (0, 5))),
    "D4": ("G2", ((0, 2, 3), (1,))),
}


def folding(source: LieType) -> Embedding:
    """Return the subalgebra fixed by a diagram symmetry: ``E6 > F4``, ``D4 > G2``."""
    if source.name not in _FOLDINGS:
        raise EasyError("type", f"no folding is recorded for {source.name}")
    name, orbits = _FOLDINGS[source.name]
    rows = [[int(j in orbit) for j in range(source.rank)] for orbit in orbits]
    return Embedding(source, lie_type(name), rows, f"{source.name} > {name} (folding)")


def _defining_weights(target: LieType, count: int) -> list[Weight]:
    first = [Fraction(0)] * target.rank
    first[0] = Fraction(1)
    found = [tuple(first)]
    for k in range(count - 1):
        root = target.simple_roots[k]
        found.append(tuple(a - b for a, b in zip(found[-1], root, strict=True)))
    return found


def via_representation(
    sub: LieType, highest: Sequence[Sequence[Number]], into: str = "auto"
) -> Embedding:
    """Return the embedding in ``sl``, ``so`` or ``sp`` given by a representation.

    ``highest`` lists the highest weights of the summands of an
    ``N``-dimensional representation ``V``.  The image lies in ``sl(N)``; it
    lies in ``so(N)`` when every summand is real and in ``sp(N)`` when every
    summand is pseudoreal, and ``into="auto"`` picks the smallest of these.
    The projection is fixed by sending the weights of the defining
    representation to the weights of ``V``.
    """
    if sub.abelian:
        raise EasyError("type", "a semisimple subalgebra is required")
    weights: list[Weight] = []
    kinds = set()
    for top in highest:
        kinds.add(sub.reality(top))
        for weight, count in sub.character(top).items():
            weights += [weight] * count
    total = len(weights)
    if into == "auto":
        into = "so" if kinds == {"real"} else "sp" if kinds == {"pseudoreal"} else "sl"
    if into == "sl":
        name, count = f"A{total - 1}", total - 1
    elif into == "so" and kinds == {"real"}:
        name, count = (f"B{total // 2}" if total % 2 else f"D{total // 2}"), total // 2
    elif into == "sp" and kinds == {"pseudoreal"}:
        name, count = f"C{total // 2}", total // 2
    else:
        raise EasyError(
            "type", f"the representation does not preserve the form of {into}({total})"
        )
    target = lie_type(name)
    weights.sort(key=lambda weight: (sub.height(weight), weight), reverse=True)
    defining = _defining_weights(target, count)
    basis = inverse(list(zip(*defining, strict=True)))
    rows = [
        [
            sum((weights[k][r] * basis[k][j] for k in range(count)), Fraction(0))
            for j in range(count)
        ]
        for r in range(sub.rank)
    ]
    first = [0] * target.rank
    first[0] = 1
    found = Embedding(target, sub, rows, f"{name} > {sub.name} (by representation)")
    projected: Character = {}
    for weight, multiplicity in target.character(first).items():
        image = found.project(weight)
        projected[image] = projected.get(image, 0) + multiplicity
    expected: Character = {}
    for weight in weights:
        expected[weight] = expected.get(weight, 0) + 1
    if projected != expected:  # pragma: no cover - guards the construction
        raise EasyError("type", "the representation does not define an embedding")
    return found


# --- explicit subalgebras ----------------------------------------------------


def from_subalgebra(
    algebra: LieAlgebra, vectors: Iterable[Sequence[Number]]
) -> Embedding:
    """Return the embedding of a subalgebra given by explicit elements.

    ``vectors`` are coordinate vectors that generate a semisimple subalgebra
    of a semisimple ``algebra``.  A Cartan subalgebra of the subalgebra is
    extended to one of the whole algebra, and the projection matrix
    expresses the coroots of the one in the coroots of the other.  Nothing
    is looked up: the rule follows from the structure constants.
    """
    sub, basis = ls.subalgebra(algebra, vectors)
    inner = ls.root_decomposition(sub)
    torus = [
        tuple(
            sum((c * basis[k][i] for k, c in enumerate(element) if c), Fraction(0))
            for i in range(algebra.dimension)
        )
        for element in inner.torus
    ]
    outer = ls.root_decomposition(algebra, torus=torus)
    if len(inner.simple) != inner.rank or len(outer.simple) != outer.rank:
        raise EasyError("semisimple", "both algebras must be semisimple")

    def coroots(found: ls.RootDecomposition) -> list[tuple[ls.Gauss, ...]]:
        return [h for _, _, h in found.chevalley_generators()]

    def lifted(vector: Sequence[ls.Gauss]) -> tuple[ls.Gauss, ...]:
        return tuple(
            sum((c * basis[k][i] for k, c in enumerate(vector) if c), ls.Gauss())
            for i in range(algebra.dimension)
        )

    def flat(vector: Sequence[ls.Gauss]) -> tuple[Fraction, ...]:
        return tuple(v.re for v in vector) + tuple(v.im for v in vector)

    span = _Span(2 * algebra.dimension)
    big = coroots(outer)
    for vector in big:
        span.add(flat(vector))
    raw = []
    for vector in coroots(inner):
        found = span.coordinates(flat(lifted(vector)))
        if found is None:  # pragma: no cover - the torus was extended
            raise EasyError("closure", "a coroot leaves the Cartan subalgebra")
        raw.append(found)
    outer_order = canonical_order(outer.cartan_matrix())
    inner_order = canonical_order(inner.cartan_matrix())
    outer_order.sort(key=lambda item: (-item[0][1], item[0][0]))
    columns = [node for _, nodes in outer_order for node in nodes]
    source = LieType([kind for kind, _ in outer_order])
    pieces = [
        (kind, [[raw[node][column] for column in columns] for node in nodes])
        for kind, nodes in inner_order
    ]
    found_embedding = _assemble(source, pieces, [])
    found_embedding.label = f"{algebra.name} > {sub.name} (explicit)"
    return found_embedding


# --- maximal subalgebras, chains and regular subalgebras ---------------------

_PRINCIPAL_MAXIMAL = {("A", 2), ("G", 2), ("F", 4), ("E", 7), ("E", 8)}


def _maximal_simple(source: LieType) -> list[Embedding]:
    family, size = source.factors[0]
    found = []
    for node, mark in enumerate(marks(source)):
        if mark in PRIMES:
            found.append(regular(source, node))
        if mark == 1:
            found.append(regular(source, node, extended=False))
    if source.name in _FOLDINGS and source.name == "E6":
        found.append(folding(source))
    if (
        (family, size) in _PRINCIPAL_MAXIMAL
        or (family == "B" and size != 3)
        or family == "C"
    ):
        found.append(principal_sl2(source))
    special: list[tuple[str, list[list[int]], str]] = []
    if family == "A" and size >= 3:
        total = size + 1
        special.append((f"B{total // 2}" if total % 2 else f"D{total // 2}", [], "sl"))
        if total % 2 == 0:
            special.append((f"C{total // 2}", [], "sl"))
    if family == "D":
        special.append((f"B{size - 1}", [[0] * (size - 1)], "so"))
    if source.name == "B3":
        special.append(("G2", [], "so"))
    for name, extra, into in special:
        try:
            sub = lie_type(name)
            first = [1] + [0] * (sub.rank - 1)
            found.append(via_representation(sub, [first, *extra], into))
        except EasyError:
            continue
    return found


def maximal_subalgebras(source: LieType) -> list[Embedding]:
    """Return maximal subalgebras, one step down, with their projections.

    For each simple factor the list holds: the regular maximal subalgebras
    (a node with a prime mark deleted from the extended diagram, or a node
    with mark 1 replaced by a ``U1``); the principal ``A1`` where it is
    maximal; ``F4`` in ``E6``; and the classical cases ``so(n)`` and
    ``sp(n)`` in ``sl(n)``, ``so(2n-1)`` in ``so(2n)`` and ``G2`` in
    ``so(7)``.  The other special subalgebras, among them tensor-product
    subalgebras of the classical algebras and most special subalgebras of
    the exceptional ones, are not listed; build them with
    :func:`via_representation` or :func:`from_subalgebra`.

    Two subalgebras of the same type that restrict the adjoint
    representation in the same way are listed once.  They are related by an
    automorphism, which may be an outer one, as for the three ``A3+U1`` in
    ``D4``.
    """
    found = []
    for position, factor in enumerate(source.factors):
        single = LieType([factor])
        adjoint = single.highest_root()
        seen = set()
        for inner in _maximal_simple(single):
            content = inner.branch(adjoint)
            # Charges are compared up to sign: a dual pair of embeddings is one entry.
            rank = inner.target.semisimple_rank
            signature = (
                inner.target.key,
                tuple(
                    sorted(
                        (w[:rank], tuple(abs(c) for c in w[rank:]), n)
                        for w, n in content.items()
                    )
                ),
            )
            if signature in seen:
                continue
            seen.add(signature)
            lifted = inner if source.is_simple else on_factor(source, position, inner)
            lifted.label = inner.label
            found.append(lifted)
    return found


def chains(source: LieType, target: LieType, max_steps: int = 4) -> list[Embedding]:
    """Return the chains of maximal subalgebras from one type down to another.

    Each result is the composite embedding; its ``label`` records the path.
    Only the subalgebras of :func:`maximal_subalgebras` are used as steps.
    """
    found: list[Embedding] = []
    seen: set[tuple[object, tuple[Row, ...]]] = set()

    def descend(current: Embedding | None, here: LieType, steps: int) -> None:
        if here.key == target.key and current is not None:
            signature = (here.key, tuple(sorted(current.projection)))
            if signature not in seen:
                seen.add(signature)
                found.append(current)
            return
        if steps == 0 or here.rank < target.rank:
            return
        if here.algebra_dimension <= target.algebra_dimension:
            return
        for step in maximal_subalgebras(here):
            path = f"{here.name} > {step.target.name}"
            composite = step if current is None else current.then(step)
            composite.label = (
                path if current is None else (f"{current.label} > {step.target.name}")
            )
            descend(composite, step.target, steps - 1)

    descend(None, source, max_steps)
    return found


def regular_subalgebras(source: LieType) -> tuple[str, ...]:
    """Return the types of the regular semisimple subalgebras.

    They are generated by Dynkin's procedure: repeatedly extend the diagram
    of one simple factor and delete a node, or delete a node outright.  The
    list is by type; two subalgebras of the same type need not be conjugate.
    """
    start = LieType(source.factors)
    found = {start.key[0]}
    layer = [start]
    while layer:
        following = []
        for current in layer:
            for position, factor in enumerate(current.factors):
                single = LieType([factor])
                rest = [f for k, f in enumerate(current.factors) if k != position]
                for extended in (True, False):
                    for node in range(single.rank + int(extended)):
                        pieces = list(
                            regular(single, node, extended=extended).target.factors
                        )
                        if not pieces + rest:
                            continue
                        smaller = LieType(pieces + rest)
                        if smaller.key[0] not in found:
                            found.add(smaller.key[0])
                            following.append(smaller)
        layer = following
    ordered = sorted(
        found,
        key=lambda key: (
            -sum(size for _, size in key),
            -LieType(key).algebra_dimension,
            key,
        ),
    )
    return tuple("+".join(f"{family}{size}" for family, size in key) for key in ordered)
