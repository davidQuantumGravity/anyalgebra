"""Weights, characters and tensor products of complex reductive Lie algebras.

A type is a sum of simple factors and ``U1`` factors, written ``"E6"`` or
``"A2+A1+U1"``.  A weight is a tuple: the Dynkin labels of each simple
factor in the Bourbaki numbering, in order, followed by one charge per
``U1``.  An irreducible representation is named by its highest weight::

    import anyalgebra.weights as aw

    E6 = aw.lie_type("E6")
    print(E6.dimension((1, 0, 0, 0, 0, 0)))  # 27
    print(E6.show(E6.tensor((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0))))

Everything is integer arithmetic on the weight lattice, so it is exact for
any representation; the cost grows with the number of weights.  The
multiplicities come from Freudenthal's formula and tensor products from
Klimyk's formula (Humphreys, "Introduction to Lie Algebras and
Representation Theory", sections 22 and 24).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from functools import cached_property

from anyalgebra.easy import EasyError
from anyalgebra.lie import RootSystem, inverse

Weight = tuple[Fraction, ...]
Character = dict[Weight, int]
Number = int | Fraction


def _weight(values: Iterable[Number]) -> Weight:
    return tuple(Fraction(value) for value in values)


def smith_normal_form(
    rows: Sequence[Sequence[int]],
) -> tuple[list[int], list[list[int]]]:
    """Return ``(d, V)`` for a square integer matrix ``A``.

    ``d`` lists the invariant factors and ``V`` is a unimodular matrix of
    column operations such that ``U A V`` is diagonal with entries ``d`` for
    some unimodular ``U``.
    """
    size = len(rows)
    a = [list(row) for row in rows]
    v = [[int(i == j) for j in range(size)] for i in range(size)]
    for t in range(size):
        while True:
            entries = [
                (abs(a[i][j]), i, j)
                for i in range(t, size)
                for j in range(t, size)
                if a[i][j]
            ]
            if not entries:
                break
            _, i, j = min(entries)
            a[t], a[i] = a[i], a[t]
            for row in (*a, *v):
                row[t], row[j] = row[j], row[t]
            pivot = a[t][t]
            dirty = False
            for i in range(t + 1, size):
                factor = a[i][t] // pivot
                if factor:
                    a[i] = [x - factor * y for x, y in zip(a[i], a[t], strict=True)]
                dirty = dirty or bool(a[i][t])
            for j in range(t + 1, size):
                factor = a[t][j] // pivot
                if factor:
                    for row in (*a, *v):
                        row[j] -= factor * row[t]
                dirty = dirty or bool(a[t][j])
            if dirty:
                continue
            spoiler = next(
                (
                    i
                    for i in range(t + 1, size)
                    for j in range(t + 1, size)
                    if a[i][j] % pivot
                ),
                None,
            )
            if spoiler is None:
                break
            a[t] = [x + y for x, y in zip(a[t], a[spoiler], strict=True)]
    return [abs(a[t][t]) for t in range(size)], v


class LieType:
    """A complex reductive Lie algebra: simple factors and ``U1`` factors."""

    def __init__(self, factors: Sequence[tuple[str, int]], abelian: int = 0) -> None:
        self.factors: tuple[tuple[str, int], ...] = tuple(factors)
        self.abelian = abelian
        self._systems = [RootSystem(family, size) for family, size in self.factors]
        self.semisimple_rank = sum(size for _, size in self.factors)
        self.rank = self.semisimple_rank + abelian
        if self.rank == 0:
            raise EasyError("type", "a type needs at least one factor")

    # description

    @property
    def name(self) -> str:
        """Return the type as text, such as ``A2+A1+U1``."""
        parts = [f"{family}{size}" for family, size in self.factors]
        return "+".join(parts + ["U1"] * self.abelian)

    @property
    def key(self) -> tuple[tuple[tuple[str, int], ...], int]:
        """Return the type with its simple factors sorted, for comparison."""
        return tuple(sorted(self.factors, key=lambda f: (-f[1], f[0]))), self.abelian

    @property
    def is_simple(self) -> bool:
        """Say whether there is one simple factor and no ``U1``."""
        return len(self.factors) == 1 and self.abelian == 0

    @property
    def algebra_dimension(self) -> int:
        """Return the dimension of the Lie algebra."""
        return sum(system.dimension for system in self._systems) + self.abelian

    def __repr__(self) -> str:
        return f"LieType({self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LieType):
            return NotImplemented
        return self.factors == other.factors and self.abelian == other.abelian

    def __hash__(self) -> int:
        return hash((self.factors, self.abelian))

    # lattice data

    @cached_property
    def offsets(self) -> tuple[int, ...]:
        """Return the position of the first label of each simple factor."""
        found, position = [], 0
        for _, size in self.factors:
            found.append(position)
            position += size
        return tuple(found)

    @cached_property
    def cartan(self) -> tuple[tuple[int, ...], ...]:
        """Return the Cartan matrix of the semisimple part, block by block."""
        size = self.semisimple_rank
        matrix = [[0] * size for _ in range(size)]
        for offset, system in zip(self.offsets, self._systems, strict=True):
            for i, row in enumerate(system.cartan):
                for j, value in enumerate(row):
                    matrix[offset + i][offset + j] = value
        return tuple(tuple(row) for row in matrix)

    @cached_property
    def inverse_cartan(self) -> tuple[tuple[Fraction, ...], ...]:
        """Return the inverse of the Cartan matrix."""
        return inverse(self.cartan) if self.semisimple_rank else ()

    @cached_property
    def lengths(self) -> tuple[Fraction, ...]:
        """Return the squared lengths of the simple roots; long roots have 2."""
        found: list[Fraction] = []
        for system in self._systems:
            raw = system._symmetrizer()
            scale = Fraction(2) / max(raw)
            found += [value * scale for value in raw]
        return tuple(found)

    @cached_property
    def positive_roots(self) -> tuple[tuple[tuple[int, ...], Weight], ...]:
        """Return each positive root as (simple-root coefficients, Dynkin labels)."""
        size = self.semisimple_rank
        found = []
        for offset, system in zip(self.offsets, self._systems, strict=True):
            for root in system.positive:
                coefficients = [0] * size
                coefficients[offset : offset + system.rank] = root
                found.append((tuple(coefficients), self._labels(coefficients)))
        return tuple(found)

    def _labels(self, coefficients: Sequence[int]) -> Weight:
        size = self.semisimple_rank
        labels = [
            sum(coefficients[i] * self.cartan[i][j] for i in range(size))
            for j in range(size)
        ]
        return _weight(labels + [0] * self.abelian)

    @cached_property
    def simple_roots(self) -> tuple[Weight, ...]:
        """Return the simple roots as weights."""
        size = self.semisimple_rank
        return tuple(
            self._labels([int(i == j) for j in range(size)]) for i in range(size)
        )

    @cached_property
    def rho(self) -> Weight:
        """Return half the sum of the positive roots: all labels one."""
        return _weight([1] * self.semisimple_rank + [0] * self.abelian)

    def check(self, weight: Sequence[Number]) -> Weight:
        """Return a weight as exact fractions after checking its length."""
        if len(weight) != self.rank:
            raise EasyError(
                "weight", f"{self.name} needs {self.rank} entries per weight"
            )
        return _weight(weight)

    def _highest(self, weight: Sequence[Number]) -> Weight:
        found = self.check(weight)
        if any(
            value < 0 or value.denominator != 1
            for value in found[: self.semisimple_rank]
        ):
            raise EasyError(
                "weight", "a highest weight has nonnegative integer Dynkin labels"
            )
        return found

    def inner(self, first: Sequence[Number], second: Sequence[Number]) -> Fraction:
        """Return the invariant inner product of two weights.

        Long roots have squared length 2 in every simple factor, and the
        ``U1`` charges multiply.
        """
        size = self.semisimple_rank
        total = Fraction(0)
        for i in range(size):
            if first[i]:
                for j in range(size):
                    if second[j] and self.inverse_cartan[i][j]:
                        total += (
                            first[i]
                            * self.inverse_cartan[i][j]
                            * self.lengths[j]
                            / 2
                            * second[j]
                        )
        for k in range(size, self.rank):
            total += Fraction(first[k]) * second[k]
        return total

    def _pair(self, weight: Weight, coefficients: Sequence[int]) -> Fraction:
        """Return the inner product of a weight with a root."""
        return sum(
            (
                c * weight[i] * self.lengths[i] / 2
                for i, c in enumerate(coefficients)
                if c
            ),
            Fraction(0),
        )

    # the Weyl group

    def reflect(self, weight: Weight, index: int) -> Weight:
        """Return the reflection of a weight in simple root ``index``."""
        factor = weight[index]
        root = self.simple_roots[index]
        return tuple(a - factor * b for a, b in zip(weight, root, strict=True))

    def to_dominant(self, weight: Sequence[Number]) -> tuple[Weight, int]:
        """Return the dominant weight of the orbit and the sign of the element."""
        current, sign = _weight(weight), 1
        while True:
            index = next(
                (i for i in range(self.semisimple_rank) if current[i] < 0), None
            )
            if index is None:
                return current, sign
            current, sign = self.reflect(current, index), -sign

    def orbit(self, weight: Sequence[Number]) -> tuple[Weight, ...]:
        """Return the Weyl group orbit of a weight."""
        start = _weight(weight)
        found = {start}
        layer = [start]
        while layer:
            following = []
            for current in layer:
                for index in range(self.semisimple_rank):
                    if current[index]:
                        image = self.reflect(current, index)
                        if image not in found:
                            found.add(image)
                            following.append(image)
            layer = following
        return tuple(sorted(found, reverse=True))

    def is_dominant(self, weight: Sequence[Number]) -> bool:
        """Say whether every Dynkin label is nonnegative."""
        return all(value >= 0 for value in weight[: self.semisimple_rank])

    def height(self, weight: Sequence[Number]) -> Fraction:
        """Return the sum of the coefficients of a weight in the simple roots."""
        size = self.semisimple_rank
        return sum(
            (
                Fraction(weight[i]) * self.inverse_cartan[i][j]
                for i in range(size)
                for j in range(size)
            ),
            Fraction(0),
        )

    # representations

    def dimension(self, highest: Sequence[Number]) -> int:
        """Return the dimension of an irreducible representation (Weyl)."""
        weight = self._highest(highest)
        shifted = tuple(a + b for a, b in zip(weight, self.rho, strict=True))
        numerator = denominator = Fraction(1)
        for coefficients, _ in self.positive_roots:
            numerator *= self._pair(shifted, coefficients)
            denominator *= self._pair(self.rho, coefficients)
        value = numerator / denominator
        assert value.denominator == 1
        return int(value)

    def dominant_character(self, highest: Sequence[Number]) -> Character:
        """Return the dominant weights of a representation with multiplicities."""
        top = self._highest(highest)
        cached = self._cache.get(top)
        if cached is not None:
            return cached
        dominant = {top}
        layer = [top]
        while layer:
            following = []
            for current in layer:
                for _, labels in self.positive_roots:
                    lower = tuple(a - b for a, b in zip(current, labels, strict=True))
                    if self.is_dominant(lower) and lower not in dominant:
                        dominant.add(lower)
                        following.append(lower)
            layer = following
        shifted = tuple(a + b for a, b in zip(top, self.rho, strict=True))
        norm = self.inner(shifted, shifted)
        found: Character = {top: 1}
        for weight in sorted(dominant, key=self.height, reverse=True):
            if weight == top:
                continue
            moved = tuple(a + b for a, b in zip(weight, self.rho, strict=True))
            denominator = norm - self.inner(moved, moved)
            total = Fraction(0)
            for coefficients, labels in self.positive_roots:
                current = weight
                while True:
                    current = tuple(a + b for a, b in zip(current, labels, strict=True))
                    multiplicity = found.get(self.to_dominant(current)[0])
                    if multiplicity is None:
                        break
                    total += 2 * multiplicity * self._pair(current, coefficients)
            value = total / denominator
            assert value.denominator == 1
            if value:
                found[weight] = int(value)
        self._cache[top] = found
        return found

    @cached_property
    def _cache(self) -> dict[Weight, Character]:
        return {}

    def character(self, highest: Sequence[Number]) -> Character:
        """Return every weight of a representation with its multiplicity."""
        found: Character = {}
        for weight, multiplicity in self.dominant_character(highest).items():
            for image in self.orbit(weight):
                found[image] = multiplicity
        return found

    def decompose(self, character: Mapping[Weight, int]) -> Character:
        """Return the irreducible content of a Weyl-invariant set of weights.

        The result maps highest weights to multiplicities.  Weights that do
        not form a character raise an error.
        """
        remaining = {
            weight: multiplicity
            for weight, multiplicity in character.items()
            if multiplicity and self.is_dominant(weight)
        }
        found: Character = {}
        while remaining:
            top = max(remaining, key=lambda weight: (self.height(weight), weight))
            count = remaining[top]
            if count < 0 or any(
                v.denominator != 1 for v in top[: self.semisimple_rank]
            ):
                raise EasyError("character", "the weights do not form a character")
            found[top] = count
            for weight, multiplicity in self.dominant_character(top).items():
                left = remaining.get(weight, 0) - count * multiplicity
                if left:
                    remaining[weight] = left
                else:
                    remaining.pop(weight, None)
        return found

    def tensor(self, first: Sequence[Number], second: Sequence[Number]) -> Character:
        """Return the irreducible content of a tensor product (Klimyk)."""
        a, b = self._highest(first), self._highest(second)
        if self.dimension(a) < self.dimension(b):
            a, b = b, a
        found: Character = {}
        for weight, multiplicity in self.character(b).items():
            moved = tuple(
                x + y + z for x, y, z in zip(a, weight, self.rho, strict=True)
            )
            dominant, sign = self.to_dominant(moved)
            if any(value == 0 for value in dominant[: self.semisimple_rank]):
                continue
            top = tuple(x - y for x, y in zip(dominant, self.rho, strict=True))
            found[top] = found.get(top, 0) + sign * multiplicity
        return {weight: count for weight, count in found.items() if count}

    def _power(self, highest: Sequence[Number], degree: int, sign: int) -> Character:
        if type(degree) is not int or degree < 0:
            raise EasyError("weight", "the degree must be a nonnegative integer")
        base = self.character(highest)
        adams = [
            {tuple(k * v for v in weight): m for weight, m in base.items()}
            for k in range(degree + 1)
        ]
        zero = _weight([0] * self.rank)
        powers: list[dict[Weight, Fraction]] = [{zero: Fraction(1)}]
        for k in range(1, degree + 1):
            total: dict[Weight, Fraction] = {}
            for i in range(1, k + 1):
                factor = Fraction(sign ** (i - 1), k)
                for u, m in adams[i].items():
                    for w, n in powers[k - i].items():
                        key = tuple(x + y for x, y in zip(u, w, strict=True))
                        total[key] = total.get(key, Fraction(0)) + factor * m * n
            powers.append(total)
        return self.decompose(
            {weight: int(value) for weight, value in powers[degree].items() if value}
        )

    def symmetric_power(self, highest: Sequence[Number], degree: int) -> Character:
        """Return the irreducible content of a symmetric power."""
        return self._power(highest, degree, 1)

    def exterior_power(self, highest: Sequence[Number], degree: int) -> Character:
        """Return the irreducible content of an exterior power."""
        return self._power(highest, degree, -1)

    # invariants of a representation

    def casimir(self, highest: Sequence[Number]) -> Fraction:
        """Return the quadratic Casimir value ``(lambda, lambda + 2 rho)``."""
        weight = self._highest(highest)
        shifted = tuple(a + 2 * b for a, b in zip(weight, self.rho, strict=True))
        return self.inner(weight, shifted)

    def index(self, highest: Sequence[Number]) -> Fraction:
        """Return the Dynkin index of a representation of a simple type.

        It is ``dim(V) * casimir / dim(g)``, which is 1 for the defining
        representation of ``A_n`` and twice the dual Coxeter number for the
        adjoint representation.
        """
        if not self.is_simple:
            raise EasyError("type", "the Dynkin index is defined per simple factor")
        return (
            Fraction(self.dimension(highest))
            * self.casimir(highest)
            / self.algebra_dimension
        )

    def highest_root(self) -> Weight:
        """Return the highest weight of the adjoint representation."""
        if not self.is_simple:
            raise EasyError("type", "the highest root needs a simple type")
        return self.positive_roots[-1][1]

    def dual(self, highest: Sequence[Number]) -> Weight:
        """Return the highest weight of the dual representation."""
        weight = self._highest(highest)
        return self.to_dominant(tuple(-value for value in weight))[0]

    def reality(self, highest: Sequence[Number]) -> str:
        """Return ``real``, ``pseudoreal`` or ``complex``."""
        weight = self._highest(highest)
        if self.dual(weight) != weight:
            return "complex"
        total = sum(
            (
                2 * self._pair(weight, coefficients) / self._pair(labels, coefficients)
                for coefficients, labels in self.positive_roots
            ),
            Fraction(0),
        )
        # total is <lambda, 2 rho^vee>, an integer whose parity decides.
        return "real" if int(total) % 2 == 0 else "pseudoreal"

    # global structure

    @cached_property
    def _smith(self) -> tuple[list[int], list[list[int]]]:
        return smith_normal_form(self.cartan)

    def center(self) -> tuple[int, ...]:
        """Return the orders of the cyclic factors of the centre.

        This is the centre of the simply connected group of the semisimple
        part, the weight lattice modulo the root lattice: ``(3,)`` for
        ``E6`` and ``(2, 2)`` for ``D4``.
        """
        return tuple(value for value in self._smith[0] if value > 1)

    def congruency(self, weight: Sequence[Number]) -> tuple[int, ...]:
        """Return the class of a weight modulo the root lattice.

        There is one entry per cyclic factor of :meth:`center`.  A weight in
        the root lattice has class zero, and the class is additive.
        """
        factors, columns = self._smith
        labels = [int(value) for value in weight[: self.semisimple_rank]]
        return tuple(
            sum(labels[i] * columns[i][k] for i in range(len(labels))) % factor
            for k, factor in enumerate(factors)
            if factor > 1
        )

    def acting_center(self, representations: Iterable[Sequence[Number]]) -> int:
        """Return the order of the part of the centre that acts nontrivially.

        The simply connected group acts on the given representations through
        its quotient by a central subgroup; the centre of that quotient has
        this order.  It is 1 exactly when every representation is one of the
        adjoint group.
        """
        factors = self.center()
        classes = {self.congruency(weight) for weight in representations}
        generated = {tuple(0 for _ in factors)}
        layer = list(generated)
        while layer:
            following = []
            for element in layer:
                for step in classes:
                    image = tuple(
                        (a + b) % n
                        for a, b, n in zip(element, step, factors, strict=True)
                    )
                    if image not in generated:
                        generated.add(image)
                        following.append(image)
            layer = following
        return len(generated)

    def fundamental_group(self, representations: Iterable[Sequence[Number]]) -> int:
        """Return the order of the fundamental group of the group that acts faithfully.

        It is the order of the centre divided by :meth:`acting_center`: 1 for
        the simply connected group and the full order for the adjoint group.
        """
        order = 1
        for value in self.center():
            order *= value
        return order // self.acting_center(representations)

    # display

    def label(self, highest: Sequence[Number]) -> str:
        """Return a short name: the dimensions per factor, then the charges.

        A complex representation whose highest weight is smaller than that of
        its dual, in the order of tuples, is marked ``*``.
        """
        weight = self._highest(highest)
        parts = []
        for offset, (family, size) in zip(self.offsets, self.factors, strict=True):
            single = LieType([(family, size)])
            piece = weight[offset : offset + size]
            star = "*" if single.dual(piece) > piece else ""
            parts.append(f"{single.dimension(piece)}{star}")
        if not parts:
            text = "1"
        else:
            text = parts[0] if len(parts) == 1 else "(" + ",".join(parts) + ")"
        charges = weight[self.semisimple_rank :]
        if charges:
            text += "(" + ",".join(str(charge) for charge in charges) + ")"
        return text

    def show(self, content: Mapping[Weight, int]) -> str:
        """Return a decomposition as text, largest representations first."""

        def size(weight: Weight) -> int:
            return self.dimension(weight)

        terms = []
        used: dict[str, int] = {}
        for weight in sorted(content, key=lambda w: (-size(w), tuple(-v for v in w))):
            count = content[weight]
            name = self.label(weight)
            # A second representation with the same name gets a prime.
            primes = used.get(name, 0)
            used[name] = primes + 1
            terms.append((f"{count} x " if count != 1 else "") + name + "'" * primes)
        return " + ".join(terms) if terms else "0"

    def total_dimension(self, content: Mapping[Weight, int]) -> int:
        """Return the dimension of a direct sum of representations."""
        return sum(self.dimension(w) * count for w, count in content.items())


def lie_type(text: str) -> LieType:
    """Return a type from text such as ``"E6"`` or ``"A2+A1+U1"``."""
    factors: list[tuple[str, int]] = []
    abelian = 0
    for part in text.replace(" ", "").split("+"):
        if part.upper() == "U1":
            abelian += 1
        elif len(part) >= 2 and part[0].upper() in "ABCDEFG" and part[1:].isdigit():
            factors.append((part[0].upper(), int(part[1:])))
        else:
            raise EasyError("type", f"{part!r} is not a factor such as A2, E6 or U1")
    return LieType(factors, abelian)
