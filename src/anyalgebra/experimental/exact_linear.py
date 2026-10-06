"""Neutral exact sparse-linear utilities for experimental constructions.

This module owns the exact rational operator type and the linear-algebra
checks shared by the experimental constructions: independence and coordinate
reduction, trace forms, exact inertia, modular rank, an exhaustive Jacobi
check for tables of structure constants, and a controlled numerical
exponential.  Nothing here depends on a particular algebra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import ceil, isfinite, lcm


def _require_exact_scalar(value: Fraction | int) -> Fraction:
    if type(value) is Fraction:
        return value
    if type(value) is int:
        return Fraction(value)
    raise TypeError("exact scalar must be an exact integer or Fraction")


def _require_float_vector(
    value: Sequence[float | Fraction | int],
) -> tuple[float, ...]:
    entries = tuple(value)
    if any(type(entry) not in {float, int, Fraction} for entry in entries):
        raise TypeError("vector entries must be floats, integers, or Fractions")
    return tuple(float(entry) for entry in entries)


def _canonical_entries(
    dimension: int, values: Mapping[tuple[int, int], Fraction | int]
) -> tuple[tuple[int, int, Fraction], ...]:
    if type(dimension) is not int or dimension < 1:
        raise ValueError("operator dimension must be a positive built-in integer")
    entries: list[tuple[int, int, Fraction]] = []
    for (row, column), raw_value in values.items():
        if type(row) is not int or type(column) is not int:
            raise TypeError("operator indices must be built-in integers")
        if not 0 <= row < dimension or not 0 <= column < dimension:
            raise ValueError("operator entry lies outside its declared dimension")
        value = _require_exact_scalar(raw_value)
        if value:
            entries.append((row, column, value))
    return tuple(sorted(entries))


_HASH_NOT_COMPUTED = -1
"""Sentinel for the lazy operator hash; ``hash()`` never returns ``-1``."""


@dataclass(frozen=True, slots=True)
class SparseRationalOperator:
    """An exact square sparse matrix with deterministic canonical storage."""

    dimension: int
    entries: tuple[tuple[int, int, Fraction], ...]
    _hash_cache: int = field(
        default=_HASH_NOT_COMPUTED, init=False, repr=False, compare=False
    )

    @classmethod
    def from_mapping(
        cls,
        dimension: int,
        values: Mapping[tuple[int, int], Fraction | int],
    ) -> SparseRationalOperator:
        return cls._from_canonical(dimension, _canonical_entries(dimension, values))

    @classmethod
    def _from_canonical(
        cls, dimension: int, entries: tuple[tuple[int, int, Fraction], ...]
    ) -> SparseRationalOperator:
        """Wrap entries that are already validated, nonzero, and sorted.

        Public construction validates in ``__post_init__``.  Arithmetic on
        validated operators produces canonical data by construction, so this
        internal path skips a second canonicalization.
        """
        operator = object.__new__(cls)
        object.__setattr__(operator, "dimension", dimension)
        object.__setattr__(operator, "entries", entries)
        object.__setattr__(operator, "_hash_cache", _HASH_NOT_COMPUTED)
        return operator

    @classmethod
    def _from_exact_cells(
        cls, dimension: int, values: Mapping[tuple[int, int], Fraction]
    ) -> SparseRationalOperator:
        """Canonicalize cells computed from already validated operators."""
        return cls._from_canonical(
            dimension,
            tuple(
                sorted(
                    (row, column, value)
                    for (row, column), value in values.items()
                    if value
                )
            ),
        )

    def __post_init__(self) -> None:
        canonical = _canonical_entries(
            self.dimension,
            {(row, column): value for row, column, value in self.entries},
        )
        if canonical != self.entries:
            raise ValueError("operator entries must be unique, nonzero, and canonical")
        # ``Fraction(2) == 2``, so the comparison above accepts equal built-in
        # integers.  Store the canonical tuple so that every stored coefficient
        # really is a Fraction and later division stays exact.
        object.__setattr__(self, "entries", canonical)

    def __hash__(self) -> int:
        # Operators serve as cache keys inside tuples of many operators.
        # Hashing every rational entry on each lookup dominated run time, so
        # the value-based hash is computed once per immutable operator.
        cached = self._hash_cache
        if cached == _HASH_NOT_COMPUTED:
            cached = hash((self.dimension, self.entries))
            object.__setattr__(self, "_hash_cache", cached)
        return cached

    def add(self, right: SparseRationalOperator) -> SparseRationalOperator:
        """Return the exact sum of two same-dimensional operators."""
        if not isinstance(right, SparseRationalOperator):
            raise TypeError("sparse operators add only to sparse operators")
        if self.dimension != right.dimension:
            raise ValueError("operator dimensions do not match")
        values = {(row, column): value for row, column, value in self.entries}
        for row, column, value in right.entries:
            key = (row, column)
            existing = values.get(key)
            values[key] = value if existing is None else existing + value
        return SparseRationalOperator._from_exact_cells(self.dimension, values)

    def scale(self, scalar: Fraction | int) -> SparseRationalOperator:
        factor = _require_exact_scalar(scalar)
        if not factor:
            return SparseRationalOperator._from_canonical(self.dimension, ())
        # A nonzero rational factor preserves both the support and its order.
        return SparseRationalOperator._from_canonical(
            self.dimension,
            tuple((row, column, factor * value) for row, column, value in self.entries),
        )

    def transpose(self) -> SparseRationalOperator:
        """Return the exact matrix transpose."""
        return SparseRationalOperator._from_canonical(
            self.dimension,
            tuple(sorted((column, row, value) for row, column, value in self.entries)),
        )

    def __matmul__(self, right: SparseRationalOperator) -> SparseRationalOperator:
        if not isinstance(right, SparseRationalOperator):
            return NotImplemented
        if self.dimension != right.dimension:
            raise ValueError("operator dimensions do not match")
        right_by_row: dict[int, list[tuple[int, Fraction]]] = {}
        for row, column, value in right.entries:
            right_by_row.setdefault(row, []).append((column, value))
        result: dict[tuple[int, int], Fraction] = {}
        for row, shared, left_value in self.entries:
            for column, right_value in right_by_row.get(shared, ()):
                key = (row, column)
                product = left_value * right_value
                existing = result.get(key)
                result[key] = product if existing is None else existing + product
        return SparseRationalOperator._from_exact_cells(self.dimension, result)

    def commutator(self, right: SparseRationalOperator) -> SparseRationalOperator:
        return (self @ right).add((right @ self).scale(-1))

    def apply(self, value: Sequence[Fraction | int]) -> tuple[Fraction, ...]:
        entries = tuple(value)
        if any(type(entry) not in {Fraction, int} for entry in entries):
            raise TypeError("vector entries must be exact integers or Fractions")
        vector = tuple(Fraction(entry) for entry in entries)
        if len(vector) != self.dimension:
            raise ValueError("vector dimension does not match operator")
        result = [Fraction() for _ in vector]
        for row, column, coefficient in self.entries:
            result[row] += coefficient * vector[column]
        return tuple(result)

    def apply_float(self, value: Sequence[float | Fraction | int]) -> tuple[float, ...]:
        """Apply the exact operator after embedding its coefficients in ``float``."""
        vector = _require_float_vector(value)
        if len(vector) != self.dimension:
            raise ValueError("vector dimension does not match operator")
        if any(not isfinite(entry) for entry in vector):
            raise ValueError("floating-point vector entries must be finite")
        result = [0.0 for _ in vector]
        for row, column, coefficient in self.entries:
            result[row] += float(coefficient) * vector[column]
        if any(not isfinite(entry) for entry in result):
            raise ArithmeticError("floating sparse-operator action became non-finite")
        return tuple(result)


def identity_sparse_operator(dimension: int) -> SparseRationalOperator:
    """Return the exact identity operator of the requested dimension."""
    return SparseRationalOperator.from_mapping(
        dimension, {(index, index): 1 for index in range(dimension)}
    )


def sparse_operator_max_norm(operator: SparseRationalOperator) -> float:
    """Return the operator norm induced by the vector maximum norm.

    This is the largest absolute row sum of the coefficients embedded in
    ``float``.
    """
    sums: dict[int, float] = {}
    for row, _, coefficient in operator.entries:
        sums[row] = sums.get(row, 0.0) + abs(float(coefficient))
    return max(sums.values(), default=0.0)


def scaled_taylor_action(
    apply: Callable[[tuple[float, ...]], Sequence[float]],
    norm: float,
    value: Sequence[float],
    *,
    parameter: float = 1.0,
    tolerance: float = 1e-15,
    max_terms: int = 256,
    max_steps: int = 100_000,
    max_step_norm: float = 1.0,
) -> tuple[float, ...]:
    """Numerically apply ``exp(parameter * A)`` when ``apply`` evaluates ``A v``.

    ``norm`` must bound the operator norm of ``A`` induced by the vector
    maximum norm.  The parameter interval is cut into equal substeps with
    ``|step| * norm <= min(max_step_norm, 1)``.  Each substep then sums a
    series whose terms decrease from the start, so no large terms cancel.

    ``tolerance`` bounds the accumulated truncation error: every substep stops
    when its a-priori remainder bound, taken relative to the maximum norm of
    the substep input, is at most ``tolerance`` divided by the number of
    substeps.  The stopping rule is therefore independent of the scale of
    ``value``.  Rounding error is not included in that bound.

    ``ArithmeticError`` is raised when more than ``max_steps`` substeps would
    be needed, when a substep does not reach its tolerance within
    ``max_terms`` terms, or when a non-finite value appears.
    """
    if type(parameter) not in {float, int, Fraction}:
        raise TypeError("exponential parameter must be a float, integer, or Fraction")
    scalar = float(parameter)
    if not isfinite(scalar):
        raise ValueError("exponential parameter must be finite")
    if type(tolerance) is not float or not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("exponential tolerance must be a positive finite float")
    if type(max_terms) is not int or max_terms < 1:
        raise ValueError("max_terms must be a positive built-in integer")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive built-in integer")
    if (
        type(max_step_norm) is not float
        or not isfinite(max_step_norm)
        or max_step_norm <= 0.0
    ):
        raise ValueError("max_step_norm must be a positive finite built-in float")
    if type(norm) is not float or not isfinite(norm) or norm < 0.0:
        raise ValueError("operator norm bound must be a finite non-negative float")
    state = _require_float_vector(value)
    if any(not isfinite(entry) for entry in state):
        raise ValueError("floating-point vector entries must be finite")

    reach = abs(scalar) * norm / min(max_step_norm, 1.0)
    if not isfinite(reach) or reach > max_steps:
        raise ArithmeticError(
            "exponential action needs more than "
            f"{max_steps} substeps for parameter-times-norm {abs(scalar) * norm!r}"
        )
    steps = max(1, ceil(reach))
    step = scalar / steps
    ratio = abs(step) * norm
    step_tolerance = tolerance / steps
    for _ in range(steps):
        if not any(state):
            break
        term = state
        total = list(state)
        bound = 1.0
        for order in range(1, max_terms + 1):
            applied = tuple(apply(term))
            if len(applied) != len(state):
                raise ValueError("operator action changed the vector dimension")
            term = tuple(step * entry / order for entry in applied)
            for index, entry in enumerate(term):
                total[index] += entry
            # After this term the neglected tail is at most
            # ratio**(order+1)/(order+1)! * 1/(1 - ratio/(order+2)).
            bound *= ratio / order
            remainder = bound * ratio / (order + 1) * (order + 2) / (order + 2 - ratio)
            if remainder <= step_tolerance:
                break
        else:
            raise ArithmeticError(
                f"exponential action did not converge within {max_terms} terms"
            )
        state = tuple(total)
        if any(not isfinite(entry) for entry in state):
            raise ArithmeticError("exponential action became non-finite")
    return state


def sparse_exponential_action(
    operator: SparseRationalOperator,
    value: Sequence[float | Fraction | int],
    *,
    parameter: float = 1.0,
    tolerance: float = 1e-15,
    max_terms: int = 256,
    max_steps: int = 100_000,
    max_step_norm: float = 1.0,
) -> tuple[float, ...]:
    """Numerically apply ``exp(parameter * operator)`` to a vector.

    This is a deliberately numerical boundary: the operator is exact and the
    result is floating point.  The work is delegated to
    :func:`scaled_taylor_action`, which sub-steps the parameter so that the
    series is free of cancellation and reports, rather than returns, a result
    that misses the requested relative tolerance.
    """
    if not isinstance(operator, SparseRationalOperator):
        raise TypeError("the exponential acts through a sparse rational operator")
    vector = _require_float_vector(value)
    if len(vector) != operator.dimension:
        raise ValueError("vector dimension does not match operator")
    return scaled_taylor_action(
        operator.apply_float,
        sparse_operator_max_norm(operator),
        vector,
        parameter=parameter,
        tolerance=tolerance,
        max_terms=max_terms,
        max_steps=max_steps,
        max_step_norm=max_step_norm,
    )


def _flatten(operator: SparseRationalOperator) -> dict[int, Fraction]:
    return {
        row * operator.dimension + column: value
        for row, column, value in operator.entries
    }


def _subtract_row(
    target: dict[int, Fraction], factor: Fraction, source: Mapping[int, Fraction]
) -> None:
    for index, value in source.items():
        updated = target.get(index, Fraction()) - factor * value
        if updated:
            target[index] = updated
        else:
            target.pop(index, None)


def independent_sparse_operators(
    candidates: Sequence[SparseRationalOperator],
) -> tuple[SparseRationalOperator, ...]:
    """Select a deterministic exact basis from a same-sized operator family."""
    family = tuple(candidates)
    if not family:
        return ()
    dimension = family[0].dimension
    if any(operator.dimension != dimension for operator in family):
        raise ValueError("all operators must have the same dimension")
    pivots: dict[int, dict[int, Fraction]] = {}
    result: list[SparseRationalOperator] = []
    for operator in family:
        row = _flatten(operator)
        while row:
            pivot = min(row)
            prior = pivots.get(pivot)
            if prior is not None:
                _subtract_row(row, row[pivot], prior)
                continue
            scale = row[pivot]
            pivots[pivot] = {index: value / scale for index, value in row.items()}
            result.append(operator)
            break
    return tuple(result)


def _coordinate_reducer(
    basis: tuple[SparseRationalOperator, ...],
) -> dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]]:
    if not basis:
        return {}
    pivots: dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]] = {}
    basis_dimension = len(basis)
    for basis_index, operator in enumerate(basis):
        row = _flatten(operator)
        coordinates = [
            Fraction(1 if index == basis_index else 0)
            for index in range(basis_dimension)
        ]
        while row:
            pivot = min(row)
            prior = pivots.get(pivot)
            if prior is not None:
                prior_row, prior_coordinates = prior
                factor = row[pivot]
                _subtract_row(row, factor, prior_row)
                coordinates = [
                    value - factor * previous
                    for value, previous in zip(
                        coordinates, prior_coordinates, strict=True
                    )
                ]
                continue
            scale = row[pivot]
            pivots[pivot] = (
                {index: value / scale for index, value in row.items()},
                tuple(value / scale for value in coordinates),
            )
            break
        else:
            raise ValueError("operator basis is linearly dependent")
    return pivots


@lru_cache(maxsize=8)
def _cached_coordinate_reducer(
    basis: tuple[SparseRationalOperator, ...],
) -> dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]]:
    return _coordinate_reducer(basis)


def sparse_operator_coordinates(
    operator: SparseRationalOperator,
    basis: Sequence[SparseRationalOperator],
) -> tuple[Fraction, ...]:
    """Return exact coordinates in an independent sparse-operator basis."""
    family = tuple(basis)
    if any(item.dimension != operator.dimension for item in family):
        raise ValueError("operator and basis dimensions do not match")
    pivots = _cached_coordinate_reducer(family)
    row = _flatten(operator)
    coordinates = [Fraction() for _ in family]
    while row:
        pivot = min(row)
        if pivot not in pivots:
            raise ValueError("operator is outside the declared span")
        pivot_row, pivot_coordinates = pivots[pivot]
        factor = row[pivot]
        _subtract_row(row, factor, pivot_row)
        coordinates = [
            value + factor * contribution
            for value, contribution in zip(coordinates, pivot_coordinates, strict=True)
        ]
    return tuple(coordinates)


def sparse_trace_gram(
    operators: Sequence[SparseRationalOperator],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the exact symmetric matrix ``Tr(A_i A_j)`` of an operator family.

    For adjoint generators this is the Killing form.  The products are
    accumulated on integer numerators over one common denominator and divided
    once per entry, which is exact and avoids rebuilding a rational lookup
    table for each of the ``n (n + 1) / 2`` pairs.
    """
    family = tuple(operators)
    if not family:
        return ()
    dimension = family[0].dimension
    if any(operator.dimension != dimension for operator in family):
        raise ValueError("all operators must have the same dimension")
    denominator = 1
    for operator in family:
        for _, _, coefficient in operator.entries:
            denominator = lcm(denominator, coefficient.denominator)
    scaled = tuple(
        tuple(
            (
                row,
                column,
                coefficient.numerator * (denominator // coefficient.denominator),
            )
            for row, column, coefficient in operator.entries
        )
        for operator in family
    )
    transposed = tuple(
        {(column, row): numerator for row, column, numerator in entries}
        for entries in scaled
    )
    square = denominator * denominator
    size = len(family)
    rows = [[Fraction() for _ in range(size)] for _ in range(size)]
    for left in range(size):
        entries = scaled[left]
        for right in range(left, size):
            lookup = transposed[right].get
            total = 0
            for row, column, numerator in entries:
                other = lookup((row, column))
                if other is not None:
                    total += numerator * other
            if total:
                entry = Fraction(total, square)
                rows[left][right] = entry
                rows[right][left] = entry
    return tuple(tuple(row) for row in rows)


SparseBracket = Sequence[tuple[int, Fraction]]


def integer_bracket_table(
    table: Sequence[Sequence[SparseBracket]],
) -> tuple[int, tuple[tuple[tuple[tuple[int, int], ...], ...], ...]]:
    """Clear denominators in a table of sparse basis brackets.

    ``table[i][j]`` lists the nonzero coordinates of ``[b_i, b_j]``.  The
    result is ``(d, scaled)`` with every scaled coefficient equal to ``d``
    times the rational one.  The table must be square, its outputs must lie in
    the basis, and it must be exactly antisymmetric; otherwise ``ValueError``
    is raised.
    """
    size = len(table)
    denominator = 1
    for row in table:
        if len(row) != size:
            raise ValueError("bracket table must be square")
        for cell in row:
            for output, coefficient in cell:
                if type(output) is not int or not 0 <= output < size:
                    raise ValueError("bracket output lies outside the basis")
                if type(coefficient) is not Fraction:
                    raise TypeError("bracket coefficients must be exact Fractions")
                denominator = lcm(denominator, coefficient.denominator)
    scaled = tuple(
        tuple(
            tuple(
                (
                    output,
                    coefficient.numerator * (denominator // coefficient.denominator),
                )
                for output, coefficient in cell
            )
            for cell in row
        )
        for row in table
    )
    for left in range(size):
        if any(value for _, value in scaled[left][left]):
            raise ValueError(f"bracket table is not alternating at index {left}")
        for right in range(left + 1, size):
            forward: dict[int, int] = {}
            for output, value in scaled[left][right]:
                forward[output] = forward.get(output, 0) + value
            for output, value in scaled[right][left]:
                forward[output] = forward.get(output, 0) + value
            if any(forward.values()):
                raise ValueError(
                    f"bracket table is not antisymmetric at ({left}, {right})"
                )
    return denominator, scaled


def jacobi_basis_triples(
    table: Sequence[Sequence[SparseBracket]],
) -> tuple[int, tuple[int, int, int] | None]:
    """Check the Jacobi identity on every distinct basis triple exactly.

    Returns ``(checked, failure)``.  ``failure`` is the first triple
    ``i < j < k`` whose Jacobiator is nonzero, and ``checked`` counts the
    triples examined before stopping, so a clean run returns
    ``(C(n, 3), None)``.

    The table is first verified to be exactly antisymmetric.  The Jacobiator
    of a bilinear antisymmetric bracket is alternating and trilinear, so the
    distinct increasing triples decide the identity for all elements.  It is
    homogeneous of degree two in the structure constants, so the test is run
    on integer numerators over one common denominator.  That is exact and
    keeps rational arithmetic out of the ``C(n, 3)`` loop.
    """
    _, scaled = integer_bracket_table(table)
    checked = 0
    for first, second, third in combinations(range(len(scaled)), 3):
        total: dict[int, int] = {}
        for left, middle, right in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            outer = scaled[left][middle]
            if not outer:
                continue
            for intermediate, inner_value in outer:
                for output, outer_value in scaled[intermediate][right]:
                    total[output] = total.get(output, 0) + inner_value * outer_value
        if total and any(total.values()):
            return checked, (first, second, third)
        checked += 1
    return checked, None


def is_prime(value: int) -> bool:
    """Deterministic Miller--Rabin test, valid for every ``value < 2**81``."""
    if value < 2:
        return False
    witnesses = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for small in witnesses:
        if value % small == 0:
            return value == small
    odd = value - 1
    twos = 0
    while odd % 2 == 0:
        odd //= 2
        twos += 1
    for witness in witnesses:
        power = pow(witness, odd, value)
        if power in {1, value - 1}:
            continue
        for _ in range(twos - 1):
            power = power * power % value
            if power == value - 1:
                break
        else:
            return False
    return True


def modular_rank(rows: Sequence[Sequence[int]], prime: int) -> int:
    """Return the rank of an integer matrix over the field of ``prime`` elements.

    Reduction modulo a prime cannot raise the rank, so the result is a lower
    bound for the rank over the rationals.
    """
    if type(prime) is not int or not is_prime(prime):
        raise ValueError("modular rank requires a built-in prime modulus")
    matrix = [[entry % prime for entry in row] for row in rows]
    width = len(matrix[0]) if matrix else 0
    if any(len(row) != width for row in matrix):
        raise ValueError("modular rank requires a rectangular matrix")
    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column], -1, prime)
        pivot_tail = [entry * inverse % prime for entry in matrix[rank][column:]]
        matrix[rank][column:] = pivot_tail
        for index in range(rank + 1, len(matrix)):
            factor = matrix[index][column]
            if factor:
                matrix[index][column:] = [
                    (entry - factor * pivot_entry) % prime
                    for entry, pivot_entry in zip(
                        matrix[index][column:], pivot_tail, strict=True
                    )
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _require_symmetric_matrix(
    matrix: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    rows = [list(row) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("inertia requires a square matrix")
    if any(type(value) is not Fraction for row in rows for value in row):
        raise TypeError("exact inertia requires Fraction entries")
    if any(
        rows[row][column] != rows[column][row]
        for row in range(size)
        for column in range(size)
    ):
        raise ValueError("inertia requires a symmetric matrix")
    return rows


def exact_symmetric_inertia(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[int, int, int]:
    """Return exact ``(positive, negative, zero)`` congruence inertia.

    Nonzero diagonal entries use one-dimensional Schur complements.  When the
    active diagonal vanishes but an off-diagonal entry remains, a nonsingular
    ``[[0,b],[b,0]]`` block contributes one positive and one negative square.
    """
    active = _require_symmetric_matrix(matrix)
    positive = 0
    negative = 0
    zero = 0
    while active:
        size = len(active)
        diagonal = next((index for index in range(size) if active[index][index]), None)
        if diagonal is not None:
            order = (
                diagonal,
                *(index for index in range(size) if index != diagonal),
            )
            arranged = [[active[row][column] for column in order] for row in order]
            pivot_row = arranged[0]
            pivot = pivot_row[0]
            if pivot > 0:
                positive += 1
            else:
                negative += 1
            # Symmetry gives arranged[column][0] == pivot_row[column].  A row
            # with a zero in the pivot column is untouched by this Schur
            # complement, and so is every column where the pivot row vanishes.
            support = tuple(column for column in range(1, size) if pivot_row[column])
            reduced: list[list[Fraction]] = []
            for row in range(1, size):
                current = arranged[row]
                factor = current[0]
                updated = current[1:]
                if factor:
                    scale = factor / pivot
                    for column in support:
                        updated[column - 1] = (
                            current[column] - scale * pivot_row[column]
                        )
                reduced.append(updated)
            active = reduced
            continue

        off_diagonal = next(
            (
                (row, column)
                for row in range(size)
                for column in range(row + 1, size)
                if active[row][column]
            ),
            None,
        )
        if off_diagonal is None:
            zero += size
            break
        first, second = off_diagonal
        order = (
            first,
            second,
            *(index for index in range(size) if index not in {first, second}),
        )
        arranged = [[active[row][column] for column in order] for row in order]
        first_row = arranged[0]
        second_row = arranged[1]
        pivot = first_row[1]
        positive += 1
        negative += 1
        hyperbolic: list[list[Fraction]] = []
        for row in range(2, size):
            current = arranged[row]
            first_factor = current[0]
            second_factor = current[1]
            if first_factor or second_factor:
                hyperbolic.append(
                    [
                        current[column]
                        - (
                            first_factor * second_row[column]
                            + second_factor * first_row[column]
                        )
                        / pivot
                        for column in range(2, size)
                    ]
                )
            else:
                hyperbolic.append(current[2:])
        active = hyperbolic
    return positive, negative, zero
