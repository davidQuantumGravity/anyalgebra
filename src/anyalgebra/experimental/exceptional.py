"""Exact Albert-algebra and compact-F4 research tools.

The coordinate convention follows the 27-coordinate Hermitian matrix layout
used by Bernardoni, Cacciatori, Cerchiai, and Scotti, while octonion products
use the named ``algmul.O.v1`` convention already shipped by AnyAlgebra.  The
two source conventions differ by an imaginary-unit relabelling and an overall
opposite multiplication; that does not change the isomorphism class.

This is an experimental API.  The elementary products are generic: given
``fractions.Fraction`` coordinates they stay exact, and given finite floats
they return floats.  Every structural predicate, basis, reducer, and
certificate requires exact ``Fraction`` input and rejects floats.
``one_parameter_action`` is the only function that turns exact input into
floating point; it returns a numerical exponential, not an exact one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from math import isfinite
from typing import TypeAlias, cast, overload

from anyalgebra.experimental.exact_linear import is_prime, scaled_taylor_action
from anyalgebra.fixtures.composition import OCTONION_POSITIVE_TRIPLES


ALBERT_DIMENSION = 27
F4_DIMENSION = 52
OCTONION_DIMENSION = 8

Scalar: TypeAlias = Fraction | float
ExactScalar: TypeAlias = Fraction | int
Octonion: TypeAlias = tuple[Scalar, ...]
ExactOctonion: TypeAlias = tuple[Fraction, ...]
AlbertElement: TypeAlias = tuple[Scalar, ...]
ExactOperator: TypeAlias = tuple[tuple[Fraction, ...], ...]
Operator: TypeAlias = tuple[tuple[Scalar, ...], ...]


@dataclass(frozen=True, slots=True)
class DerivationDimensionCertificate:
    """Two-sided exact rank certificate for ``Der(H_3(O))``."""

    unknowns: int
    nonzero_constraints: int
    prime: int
    modular_rank: int
    rational_rank: int
    nullity: int
    inner_derivation_dimension: int


def _fraction(value: int = 0) -> Fraction:
    return Fraction(value)


def _basis_vector(dimension: int, index: int) -> tuple[Fraction, ...]:
    return tuple(
        _fraction(1 if position == index else 0) for position in range(dimension)
    )


def _require_vector(
    value: Sequence[Scalar | int], dimension: int, name: str
) -> tuple[Scalar, ...]:
    result = tuple(value)
    if len(result) != dimension:
        raise ValueError(f"{name} must have exactly {dimension} coordinates")
    if all(type(entry) is Fraction for entry in result):
        return result
    for entry in result:
        kind = type(entry)
        if kind is float:
            if not isfinite(entry):
                raise ValueError(f"{name} coordinates must be finite")
        elif kind is not Fraction and kind is not int:
            raise TypeError(
                f"{name} coordinates must be exact Fractions, ints, or floats"
            )
    return tuple(Fraction(entry) if type(entry) is int else entry for entry in result)


@lru_cache(maxsize=1)
def octonion_basis() -> tuple[ExactOctonion, ...]:
    """Return ``(1,e1,...,e7)`` in the exact ``algmul.O.v1`` convention."""
    return tuple(_basis_vector(OCTONION_DIMENSION, index) for index in range(8))


def octonion_basis_product(left: int, right: int) -> tuple[int, int]:
    """Return ``(sign, basis_index)`` for one named-convention basis product."""
    if (
        type(left) is not int
        or type(right) is not int
        or not 0 <= left < 8
        or not 0 <= right < 8
    ):
        raise ValueError("octonion basis indices must be built-in integers in 0..7")
    return _octonion_product_table()[left][right]


def _derive_octonion_basis_product(left: int, right: int) -> tuple[int, int]:
    if left == 0:
        return 1, right
    if right == 0:
        return 1, left
    if left == right:
        return -1, 0
    for first, second, third in OCTONION_POSITIVE_TRIPLES:
        for input_left, input_right, output in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            if (left, right) == (input_left, input_right):
                return 1, output
            if (left, right) == (input_right, input_left):
                return -1, output
    raise AssertionError("the named Fano convention omitted an imaginary pair")


@lru_cache(maxsize=1)
def _octonion_product_table() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Tabulate the 64 basis products of the named convention once."""
    return tuple(
        tuple(_derive_octonion_basis_product(left, right) for right in range(8))
        for left in range(8)
    )


@overload
def octonion_multiply(
    left: Sequence[ExactScalar], right: Sequence[ExactScalar]
) -> ExactOctonion: ...


@overload
def octonion_multiply(left: Sequence[Scalar], right: Sequence[Scalar]) -> Octonion: ...


def octonion_multiply(
    left: Sequence[Scalar | int], right: Sequence[Scalar | int]
) -> Octonion:
    """Multiply two octonions bilinearly with explicit nonassociative order."""
    a = _require_vector(left, 8, "left octonion")
    b = _require_vector(right, 8, "right octonion")
    table = _octonion_product_table()
    result: list[Scalar] = [_fraction() for _ in range(8)]
    for i, left_coefficient in enumerate(a):
        if left_coefficient == 0:
            continue
        row = table[i]
        for j, right_coefficient in enumerate(b):
            if right_coefficient == 0:
                continue
            sign, output = row[j]
            result[output] += sign * left_coefficient * right_coefficient
    return tuple(result)


def octonion_conjugate(value: Sequence[Scalar]) -> Octonion:
    """Apply standard octonion conjugation in the named basis."""
    item = _require_vector(value, 8, "octonion")
    return (item[0], *(-coefficient for coefficient in item[1:]))


def octonion_real_part(value: Sequence[Scalar]) -> Scalar:
    return _require_vector(value, 8, "octonion")[0]


@lru_cache(maxsize=1)
def albert_basis() -> tuple[AlbertElement, ...]:
    """Return the 27 coordinate vectors for ``H_3(O)``."""
    return tuple(_basis_vector(ALBERT_DIMENSION, index) for index in range(27))


def albert_identity() -> AlbertElement:
    """Return the 3-by-3 identity in Albert coordinates."""
    return tuple(_fraction(1 if index in {0, 17, 26} else 0) for index in range(27))


def _zero_octonion(zero: Scalar = Fraction()) -> Octonion:
    return (zero,) * 8


def _albert_matrix(value: Sequence[Scalar]) -> tuple[tuple[Octonion, ...], ...]:
    coordinates = _require_vector(value, 27, "Albert element")
    zero = coordinates[0] * 0
    o12 = tuple(coordinates[1:9])
    o13 = tuple(coordinates[9:17])
    o23 = tuple(coordinates[18:26])

    def real(number: Scalar) -> Octonion:
        return (number, *(zero for _ in range(7)))

    return (
        (real(coordinates[0]), o12, o13),
        (octonion_conjugate(o12), real(coordinates[17]), o23),
        (octonion_conjugate(o13), octonion_conjugate(o23), real(coordinates[26])),
    )


def _matrix_albert(matrix: tuple[tuple[Octonion, ...], ...]) -> AlbertElement:
    return (
        matrix[0][0][0],
        *matrix[0][1],
        *matrix[0][2],
        matrix[1][1][0],
        *matrix[1][2],
        matrix[2][2][0],
    )


def _octonion_add(left: Octonion, right: Octonion) -> Octonion:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _octonion_scale(scalar: Scalar, value: Octonion) -> Octonion:
    return tuple(scalar * coefficient for coefficient in value)


def _octonionic_matrix_product(
    left: tuple[tuple[Octonion, ...], ...],
    right: tuple[tuple[Octonion, ...], ...],
) -> tuple[tuple[Octonion, ...], ...]:
    rows: list[tuple[Octonion, ...]] = []
    for i in range(3):
        row: list[Octonion] = []
        for j in range(3):
            total = _zero_octonion(left[0][0][0] * 0)
            for k in range(3):
                total = _octonion_add(total, octonion_multiply(left[i][k], right[k][j]))
            row.append(total)
        rows.append(tuple(row))
    return tuple(rows)


def albert_jordan_product(
    left: Sequence[Scalar], right: Sequence[Scalar]
) -> AlbertElement:
    """Return ``(XY+YX)/2`` with every octonion product explicitly bracketed."""
    xy = _octonionic_matrix_product(_albert_matrix(left), _albert_matrix(right))
    yx = _octonionic_matrix_product(_albert_matrix(right), _albert_matrix(left))
    half = Fraction(1, 2)
    result = tuple(
        tuple(
            _octonion_scale(half, _octonion_add(xy[i][j], yx[i][j])) for j in range(3)
        )
        for i in range(3)
    )
    return _matrix_albert(result)


def albert_trace(value: Sequence[Scalar]) -> Scalar:
    item = _require_vector(value, 27, "Albert element")
    return item[0] + item[17] + item[26]


def albert_trace_form(left: Sequence[Scalar], right: Sequence[Scalar]) -> Scalar:
    """Return the positive trace form ``tr(X o Y)`` on compact ``H_3(O)``."""
    return albert_trace(albert_jordan_product(left, right))


def _vector_add(left: Sequence[Scalar], right: Sequence[Scalar]) -> AlbertElement:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_scale(scalar: Scalar, value: Sequence[Scalar]) -> AlbertElement:
    return tuple(scalar * coefficient for coefficient in value)


def albert_freudenthal_product(
    left: Sequence[Scalar], right: Sequence[Scalar]
) -> AlbertElement:
    """Return the polarized Freudenthal product on the Albert algebra."""
    x = _require_vector(left, 27, "left Albert element")
    y = _require_vector(right, 27, "right Albert element")
    xy = albert_jordan_product(x, y)
    tr_x, tr_y, tr_xy = albert_trace(x), albert_trace(y), albert_trace(xy)
    result = _vector_add(
        xy,
        _vector_scale(
            Fraction(-1, 2), _vector_add(_vector_scale(tr_x, y), _vector_scale(tr_y, x))
        ),
    )
    identity_term = Fraction(1, 2) * (tr_x * tr_y - tr_xy)
    return _vector_add(result, _vector_scale(identity_term, albert_identity()))


def albert_determinant(value: Sequence[Scalar]) -> Scalar:
    """Return the cubic norm ``tr(X o (X x X))/3``."""
    item = _require_vector(value, 27, "Albert element")
    cross = albert_freudenthal_product(item, item)
    return albert_trace(albert_jordan_product(item, cross)) / 3


def jordan_associator(
    left: Sequence[Scalar], middle: Sequence[Scalar], right: Sequence[Scalar]
) -> AlbertElement:
    """Return the ``(left,middle,right)`` Jordan associator."""
    return _vector_add(
        albert_jordan_product(albert_jordan_product(left, middle), right),
        _vector_scale(
            -1, albert_jordan_product(left, albert_jordan_product(middle, right))
        ),
    )


def left_multiplication_operator(value: Sequence[Scalar]) -> Operator:
    """Return the 27-by-27 coordinate matrix of ``Y -> X o Y``."""
    columns = tuple(albert_jordan_product(value, basis) for basis in albert_basis())
    return tuple(tuple(column[row] for column in columns) for row in range(27))


def _require_operator(value: Sequence[Sequence[Scalar]]) -> Operator:
    rows = tuple(_require_vector(row, 27, "operator row") for row in value)
    if len(rows) != 27:
        raise ValueError("operator must have exactly 27 rows")
    return rows


def apply_operator(
    operator: Sequence[Sequence[Scalar]], value: Sequence[Scalar]
) -> AlbertElement:
    matrix = _require_operator(operator)
    vector = _require_vector(value, 27, "Albert element")
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(27)),
            vector[0] * 0,
        )
        for row in range(27)
    )


def matrix_commutator(
    left: Sequence[Sequence[Scalar]], right: Sequence[Sequence[Scalar]]
) -> Operator:
    a, b = _require_operator(left), _require_operator(right)
    zero = a[0][0] * 0
    rows: list[list[Scalar]] = [[zero for _ in range(27)] for _ in range(27)]
    nonzero_a = tuple(
        tuple((j, value) for j, value in enumerate(row) if value) for row in a
    )
    nonzero_b = tuple(
        tuple((j, value) for j, value in enumerate(row) if value) for row in b
    )
    for i in range(27):
        for k, value in nonzero_a[i]:
            for j, right_value in nonzero_b[k]:
                rows[i][j] += value * right_value
        for k, value in nonzero_b[i]:
            for j, right_value in nonzero_a[k]:
                rows[i][j] -= value * right_value
    return tuple(tuple(row) for row in rows)


def _sparse_operator(operator: Sequence[Sequence[Scalar]]) -> dict[int, Fraction]:
    matrix = _require_operator(operator)
    result: dict[int, Fraction] = {}
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if type(value) is not Fraction:
                raise TypeError("exact operator reduction requires Fraction entries")
            if value:
                result[i * 27 + j] = value
    return result


def _subtract_sparse(
    target: dict[int, Fraction], scalar: Fraction, row: dict[int, Fraction]
) -> None:
    for index, value in row.items():
        updated = target.get(index, _fraction()) - scalar * value
        if updated:
            target[index] = updated
        else:
            target.pop(index, None)


def _independent_operators(
    candidates: Sequence[ExactOperator],
) -> tuple[ExactOperator, ...]:
    pivots: dict[int, dict[int, Fraction]] = {}
    result: list[ExactOperator] = []
    for operator in candidates:
        row = _sparse_operator(operator)
        while row:
            pivot = min(row)
            existing = pivots.get(pivot)
            if existing is not None:
                _subtract_sparse(row, row[pivot], existing)
                continue
            scale = row[pivot]
            pivots[pivot] = {index: value / scale for index, value in row.items()}
            result.append(operator)
            break
    return tuple(result)


@lru_cache(maxsize=1)
def _left_basis_operators() -> tuple[ExactOperator, ...]:
    return tuple(
        cast(ExactOperator, left_multiplication_operator(value))
        for value in albert_basis()
    )


@lru_cache(maxsize=1)
def f4_inner_derivation_basis() -> tuple[ExactOperator, ...]:
    """Return 52 independent ``[L_X,L_Y]`` derivations in deterministic order."""
    left = _left_basis_operators()
    candidates: tuple[ExactOperator, ...] = tuple(
        cast(ExactOperator, matrix_commutator(left[i], left[j]))
        for i in range(27)
        for j in range(i + 1, 27)
    )
    basis = _independent_operators(candidates)
    if len(basis) != F4_DIMENSION:
        raise ArithmeticError(
            f"Albert inner-derivation span has dimension {len(basis)}, expected 52"
        )
    return basis


@lru_cache(maxsize=1)
def _jordan_basis_table() -> tuple[tuple[AlbertElement, ...], ...]:
    basis = albert_basis()
    return tuple(
        tuple(albert_jordan_product(left, right) for right in basis) for left in basis
    )


@lru_cache(maxsize=1)
def _sparse_jordan_basis_table() -> tuple[
    tuple[tuple[tuple[int, Fraction], ...], ...], ...
]:
    table = tuple(
        tuple(
            tuple(
                (index, cast(Fraction, value))
                for index, value in enumerate(product)
                if value
            )
            for product in row
        )
        for row in _jordan_basis_table()
    )
    return table


def _accumulate(
    target: dict[int, Fraction],
    source: Sequence[tuple[int, Fraction]],
    scale: Fraction,
) -> None:
    if not scale:
        return
    for index, value in source:
        updated = target.get(index, _fraction()) + scale * value
        if updated:
            target[index] = updated
        else:
            target.pop(index, None)


def is_albert_derivation(operator: Sequence[Sequence[Scalar]]) -> bool:
    """Check the derivation identity on all ordered Albert basis pairs exactly."""
    matrix = _require_operator(operator)
    if any(type(value) is not Fraction for row in matrix for value in row):
        raise TypeError("exact derivation checking requires Fraction entries")
    table = _sparse_jordan_basis_table()
    columns: tuple[tuple[tuple[int, Fraction], ...], ...] = tuple(
        tuple(
            (row, cast(Fraction, matrix[row][column]))
            for row in range(27)
            if matrix[row][column]
        )
        for column in range(27)
    )
    for i in range(27):
        for j in range(27):
            left: dict[int, Fraction] = {}
            right: dict[int, Fraction] = {}
            for product_index, coefficient in table[i][j]:
                _accumulate(left, columns[product_index], coefficient)
            for input_index, coefficient in columns[i]:
                _accumulate(right, table[input_index][j], coefficient)
            for input_index, coefficient in columns[j]:
                _accumulate(right, table[i][input_index], coefficient)
            if left != right:
                return False
    return True


def _derivation_constraint_rows() -> tuple[dict[int, int], ...]:
    """Build the integer-scaled linear system for all derivations."""
    table = _sparse_jordan_basis_table()
    rows: list[dict[int, int]] = []
    for i in range(27):
        for j in range(i, 27):
            product = table[i][j]
            for output in range(27):
                rational_row: dict[int, Fraction] = {}

                def add(
                    variable: int,
                    value: Fraction,
                    target: dict[int, Fraction] = rational_row,
                ) -> None:
                    updated = target.get(variable, _fraction()) + value
                    if updated:
                        target[variable] = updated
                    else:
                        target.pop(variable, None)

                for input_index, coefficient in product:
                    add(output * 27 + input_index, coefficient)
                for source_index in range(27):
                    for target_index, coefficient in table[source_index][j]:
                        if target_index == output:
                            add(source_index * 27 + i, -coefficient)
                    for target_index, coefficient in table[i][source_index]:
                        if target_index == output:
                            add(source_index * 27 + j, -coefficient)
                if rational_row:
                    integer_row = {
                        variable: int(coefficient * 2)
                        for variable, coefficient in rational_row.items()
                    }
                    if any(
                        Fraction(value, 2) != rational_row[key]
                        for key, value in integer_row.items()
                    ):
                        raise ArithmeticError(
                            "Albert structure coefficient denominator exceeded two"
                        )
                    rows.append(integer_row)
    return tuple(rows)


def _modular_rank(rows: Sequence[dict[int, int]], prime: int) -> int:
    pivots: dict[int, dict[int, int]] = {}
    for source in rows:
        row = {index: value % prime for index, value in source.items() if value % prime}
        while row:
            pivot = min(row)
            existing = pivots.get(pivot)
            if existing is not None:
                factor = row[pivot]
                for index, value in existing.items():
                    updated = (row.get(index, 0) - factor * value) % prime
                    if updated:
                        row[index] = updated
                    else:
                        row.pop(index, None)
                continue
            inverse = pow(row[pivot], -1, prime)
            pivots[pivot] = {
                index: value * inverse % prime for index, value in row.items()
            }
            break
    return len(pivots)


@lru_cache(maxsize=4)
def albert_derivation_dimension_certificate(
    prime: int = 1_000_003,
) -> DerivationDimensionCertificate:
    """Certify dimension 52 without solving a 10k-by-729 rational system.

    A rank of 677 after reduction modulo a prime is a lower bound for the
    rational rank.  The independently constructed 52-dimensional exact inner
    derivation space is a lower bound for the rational nullity, hence an upper
    bound of 677 for the rational rank.  Matching bounds certify both rank and
    nullity over the rationals.

    Both halves are checked here.  ``prime`` must really be an odd prime, since
    the lower bound uses arithmetic in the field of that order.  Each of the 52
    inner derivations is substituted into every constraint row, so the upper
    bound rests on a verified solution space of the same linear system.
    """
    if type(prime) is not int or prime <= 2 or prime >= 2**81:
        raise ValueError("prime must be an odd built-in prime below 2**81")
    if not is_prime(prime):
        raise ValueError("prime must be an odd built-in prime below 2**81")
    rows = _derivation_constraint_rows()
    modular_rank = _modular_rank(rows, prime)
    inner_basis = f4_inner_derivation_basis()
    for operator in inner_basis:
        unknowns = _sparse_operator(operator)
        for row in rows:
            if sum(
                coefficient * value
                for variable, coefficient in row.items()
                if (value := unknowns.get(variable)) is not None
            ):
                raise ArithmeticError(
                    "an inner derivation violates the derivation constraint system"
                )
    inner_dimension = len(inner_basis)
    upper_rank = 27 * 27 - inner_dimension
    if modular_rank != upper_rank:
        raise ArithmeticError(
            "modular lower rank and inner-derivation upper rank do not meet"
        )
    return DerivationDimensionCertificate(
        unknowns=27 * 27,
        nonzero_constraints=len(rows),
        prime=prime,
        modular_rank=modular_rank,
        rational_rank=upper_rank,
        nullity=inner_dimension,
        inner_derivation_dimension=inner_dimension,
    )


@lru_cache(maxsize=1)
def _trace_metric() -> tuple[tuple[Fraction, ...], ...]:
    basis = albert_basis()
    return tuple(
        tuple(cast(Fraction, albert_trace_form(left, right)) for right in basis)
        for left in basis
    )


def is_trace_form_skew(operator: Sequence[Sequence[Scalar]]) -> bool:
    """Check ``D^T G + G D = 0`` for the positive Albert trace metric."""
    matrix = _require_operator(operator)
    metric = _trace_metric()
    for i in range(27):
        for j in range(27):
            value = sum(
                (
                    matrix[k][i] * metric[k][j] + metric[i][k] * matrix[k][j]
                    for k in range(27)
                ),
                _fraction(),
            )
            if value != 0:
                return False
    return True


def trace_operator_gram(
    operators: Sequence[ExactOperator],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return ``-Tr(D_i D_j)`` for an exact family of compact generators."""
    family = tuple(
        cast(ExactOperator, _require_operator(operator)) for operator in operators
    )
    if any(
        type(value) is not Fraction
        for operator in family
        for row in operator
        for value in row
    ):
        raise TypeError("trace Gram construction requires Fraction entries")
    return tuple(
        tuple(
            -sum(
                (
                    left[row][column] * right[column][row]
                    for row in range(27)
                    for column in range(27)
                ),
                _fraction(),
            )
            for right in family
        )
        for left in family
    )


def positive_definite_ldl_pivots(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[Fraction, ...]:
    """Return exact positive ``D`` pivots in an ``LDL^T`` factorization."""
    rows = tuple(tuple(row) for row in matrix)
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("positive-definiteness check requires a square matrix")
    if any(type(value) is not Fraction for row in rows for value in row):
        raise TypeError("positive-definiteness check requires Fraction entries")
    if any(rows[i][j] != rows[j][i] for i in range(size) for j in range(size)):
        raise ValueError("positive-definiteness check requires a symmetric matrix")
    lower = [[_fraction() for _ in range(size)] for _ in range(size)]
    diagonal: list[Fraction] = []
    for i in range(size):
        lower[i][i] = _fraction(1)
        pivot = rows[i][i] - sum(
            (lower[i][k] * lower[i][k] * diagonal[k] for k in range(i)),
            _fraction(),
        )
        if pivot <= 0:
            raise ValueError("matrix is not positive definite")
        diagonal.append(pivot)
        for j in range(i + 1, size):
            numerator = rows[j][i] - sum(
                (lower[j][k] * lower[i][k] * diagonal[k] for k in range(i)),
                _fraction(),
            )
            lower[j][i] = numerator / pivot
    return tuple(diagonal)


def _operator_reducer(
    basis: tuple[ExactOperator, ...],
) -> dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]]:
    pivots: dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]] = {}
    dimension = len(basis)
    for basis_index, operator in enumerate(basis):
        row = _sparse_operator(operator)
        coordinates = list(_basis_vector(dimension, basis_index))
        while row:
            pivot = min(row)
            existing = pivots.get(pivot)
            if existing is not None:
                existing_row, existing_coordinates = existing
                factor = row[pivot]
                _subtract_sparse(row, factor, existing_row)
                coordinates = [
                    value - factor * previous
                    for value, previous in zip(
                        coordinates, existing_coordinates, strict=True
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


@lru_cache(maxsize=2)
def _cached_operator_reducer(
    basis: tuple[ExactOperator, ...],
) -> dict[int, tuple[dict[int, Fraction], tuple[Fraction, ...]]]:
    return _operator_reducer(basis)


@lru_cache(maxsize=1)
def _f4_operator_reducer() -> dict[
    int, tuple[dict[int, Fraction], tuple[Fraction, ...]]
]:
    return _operator_reducer(f4_inner_derivation_basis())


def operator_coordinates(
    operator: Sequence[Sequence[Scalar]], basis: Sequence[ExactOperator]
) -> tuple[Fraction, ...]:
    """Return exact coordinates in an independent operator basis or raise."""
    family = tuple(basis)
    pivots = (
        _f4_operator_reducer()
        if family is f4_inner_derivation_basis()
        else _cached_operator_reducer(family)
    )
    row = _sparse_operator(operator)
    coordinates = [_fraction() for _ in family]
    while row:
        pivot = min(row)
        if pivot not in pivots:
            raise ValueError("operator is outside the declared span")
        pivot_row, pivot_coordinates = pivots[pivot]
        factor = row[pivot]
        _subtract_sparse(row, factor, pivot_row)
        coordinates = [
            value + factor * contribution
            for value, contribution in zip(coordinates, pivot_coordinates, strict=True)
        ]
    return tuple(coordinates)


def one_parameter_action(
    generator: Sequence[Sequence[Scalar]],
    value: Sequence[Scalar],
    parameter: float,
    *,
    terms: int = 32,
) -> tuple[float, ...]:
    """Numerically apply ``exp(parameter*D)`` to an Albert element.

    The parameter interval is cut into substeps on which the Taylor series
    converges without cancellation, and ``terms`` bounds the series length of
    one substep.  ``ArithmeticError`` is raised if a substep cannot reach
    double-precision accuracy within ``terms`` terms.
    """
    if type(parameter) is not float:
        raise TypeError("parameter must be a built-in float")
    if not isfinite(parameter):
        raise ValueError("parameter must be finite")
    if type(terms) is not int or terms < 1:
        raise ValueError("terms must be a positive built-in integer")
    matrix = tuple(
        tuple(float(entry) for entry in row) for row in _require_operator(generator)
    )
    start = tuple(
        float(entry) for entry in _require_vector(value, 27, "Albert element")
    )
    sparse_rows = tuple(
        tuple((column, entry) for column, entry in enumerate(row) if entry)
        for row in matrix
    )

    def apply(vector: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            sum(entry * vector[column] for column, entry in row) for row in sparse_rows
        )

    return scaled_taylor_action(
        apply,
        max(sum(abs(entry) for entry in row) for row in matrix),
        start,
        parameter=parameter,
        tolerance=1e-16,
        max_terms=terms,
    )


def nested_left_action(
    factors: Sequence[Octonion],
    value: Octonion,
    multiply: Callable[[Octonion, Octonion], Octonion],
) -> Octonion:
    """Match AlgMul ``NestMul``: ``a1*(a2*(...*(an*x)))``."""
    result = value
    for factor in reversed(tuple(factors)):
        result = multiply(factor, result)
    return result


def nested_right_action(
    factors: Sequence[Octonion],
    value: Octonion,
    multiply: Callable[[Octonion, Octonion], Octonion],
) -> Octonion:
    """Match AlgMul ``NestMulR``: ``(((x*an)*...)*a2)*a1``."""
    result = value
    for factor in reversed(tuple(factors)):
        result = multiply(result, factor)
    return result


def nested_difference_action(
    factors: Sequence[Octonion],
    value: Octonion,
    multiply: Callable[[Octonion, Octonion], Octonion],
) -> Octonion:
    """Match legacy ``NestMulD = -NestMul-NestMulR`` exactly."""
    left = nested_left_action(factors, value, multiply)
    right = nested_right_action(factors, value, multiply)
    return tuple(-a - b for a, b in zip(left, right, strict=True))
