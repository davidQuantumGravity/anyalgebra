"""Exact degree-four Baker--Campbell--Hausdorff derivation tools.

The symbolic certificate computes ``log(exp(X) exp(Y))`` in the truncated
free associative algebra over the words in ``X`` and ``Y``.  It then solves
for the coordinates of each homogeneous component in a fixed Hall-type basis
of the free Lie algebra.  This independently exposes zero as well as nonzero
coefficients instead of assuming the familiar printed formula.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from itertools import product
from math import factorial

Scalar = Fraction | float
Vector = tuple[Scalar, ...]
Bracket = Callable[[Sequence[Scalar], Sequence[Scalar]], Vector]
Polynomial = dict[str, Fraction]


@dataclass(frozen=True, slots=True)
class BCHHallTerm:
    """One coefficient in the selected Hall-type free-Lie basis."""

    degree: int
    expression: str
    coefficient: Fraction
    x_count: int
    y_count: int


@dataclass(frozen=True, slots=True)
class BCHExactCertificate:
    """Exact receipt for the associative-log to free-Lie reconstruction."""

    maximum_degree: int
    free_lie_dimensions: tuple[int, ...]
    homogeneous_nonzero_hall_terms: tuple[int, ...]
    associative_words_checked: int
    hall_basis_terms_checked: int
    reconstruction_exact: bool


@dataclass(frozen=True, slots=True)
class _LieNode:
    symbol: str | None = None
    left: _LieNode | None = None
    right: _LieNode | None = None


_X = _LieNode(symbol="X")
_Y = _LieNode(symbol="Y")


def _bracket(left: _LieNode, right: _LieNode) -> _LieNode:
    return _LieNode(left=left, right=right)


_XY = _bracket(_X, _Y)
_HALL_BASIS: tuple[tuple[int, str, _LieNode], ...] = (
    (1, "X", _X),
    (1, "Y", _Y),
    (2, "[X,Y]", _XY),
    (3, "[X,[X,Y]]", _bracket(_X, _XY)),
    (3, "[Y,[Y,X]]", _bracket(_Y, _bracket(_Y, _X))),
    (4, "[X,[X,[X,Y]]]", _bracket(_X, _bracket(_X, _XY))),
    (4, "[Y,[X,[X,Y]]]", _bracket(_Y, _bracket(_X, _XY))),
    (4, "[Y,[Y,[Y,X]]]", _bracket(_Y, _bracket(_Y, _bracket(_Y, _X)))),
)


def _validate_degree(degree: int) -> int:
    if type(degree) is not int or not 1 <= degree <= 4:
        raise ValueError("degree must be a built-in integer in 1..4")
    return degree


def _add_polynomial(
    target: Polynomial, source: Mapping[str, Fraction], scale: Fraction = Fraction(1)
) -> None:
    for word, coefficient in source.items():
        value = target.get(word, Fraction()) + scale * coefficient
        if value:
            target[word] = value
        else:
            target.pop(word, None)


def _multiply_polynomials(
    left: Mapping[str, Fraction],
    right: Mapping[str, Fraction],
    maximum_degree: int,
) -> Polynomial:
    result: Polynomial = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word = left_word + right_word
            if len(word) <= maximum_degree:
                result[word] = result.get(word, Fraction()) + (
                    left_coefficient * right_coefficient
                )
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def _letter_exponential(letter: str, maximum_degree: int) -> Polynomial:
    return {
        letter * degree: Fraction(1, factorial(degree))
        for degree in range(maximum_degree + 1)
    }


@lru_cache(maxsize=4)
def _associative_bch(maximum_degree: int) -> tuple[tuple[str, Fraction], ...]:
    maximum_degree = _validate_degree(maximum_degree)
    exponential_product = _multiply_polynomials(
        _letter_exponential("X", maximum_degree),
        _letter_exponential("Y", maximum_degree),
        maximum_degree,
    )
    increment = dict(exponential_product)
    increment[""] -= 1
    if not increment[""]:
        del increment[""]

    logarithm: Polynomial = {}
    power: Polynomial = {"": Fraction(1)}
    for order in range(1, maximum_degree + 1):
        power = _multiply_polynomials(power, increment, maximum_degree)
        _add_polynomial(
            logarithm,
            power,
            Fraction(1 if order % 2 else -1, order),
        )
    return tuple(sorted(logarithm.items(), key=lambda item: (len(item[0]), item[0])))


def bch_associative_word_coefficients(
    degree: int = 4,
) -> tuple[tuple[str, Fraction], ...]:
    """Derive exact coefficients of ``log(exp(X) exp(Y))`` as words."""
    return _associative_bch(_validate_degree(degree))


@cache
def _expand_lie(node: _LieNode) -> tuple[tuple[str, Fraction], ...]:
    if node.symbol is not None:
        return ((node.symbol, Fraction(1)),)
    if node.left is None or node.right is None:
        raise ArithmeticError("malformed internal Lie expression")
    left = dict(_expand_lie(node.left))
    right = dict(_expand_lie(node.right))
    degree = max(len(word) for word in left) + max(len(word) for word in right)
    result = _multiply_polynomials(left, right, degree)
    _add_polynomial(result, _multiply_polynomials(right, left, degree), Fraction(-1))
    return tuple(sorted(result.items()))


def _words(degree: int) -> tuple[str, ...]:
    return tuple("".join(letters) for letters in product("XY", repeat=degree))


def _coordinates_in_basis(
    target: Mapping[str, Fraction], basis: Sequence[_LieNode], degree: int
) -> tuple[Fraction, ...]:
    words = _words(degree)
    expansions = tuple(dict(_expand_lie(node)) for node in basis)
    rows = [
        [expansion.get(word, Fraction()) for expansion in expansions]
        + [target.get(word, Fraction())]
        for word in words
    ]
    pivot_rows: list[int] = []
    row = 0
    for column in range(len(basis)):
        pivot = next(
            (
                candidate
                for candidate in range(row, len(rows))
                if rows[candidate][column]
            ),
            None,
        )
        if pivot is None:
            raise ArithmeticError("declared Hall basis is linearly dependent")
        rows[row], rows[pivot] = rows[pivot], rows[row]
        pivot_value = rows[row][column]
        rows[row] = [entry / pivot_value for entry in rows[row]]
        for other in range(len(rows)):
            if other == row or not rows[other][column]:
                continue
            factor = rows[other][column]
            rows[other] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[other], rows[row], strict=True)
            ]
        pivot_rows.append(row)
        row += 1

    if any(not any(values[:-1]) and values[-1] for values in rows):
        raise ArithmeticError("associative BCH component is outside the Lie basis span")
    coordinates = tuple(rows[pivot_row][-1] for pivot_row in pivot_rows)
    reconstructed: Polynomial = {}
    for coordinate, expansion in zip(coordinates, expansions, strict=True):
        _add_polynomial(reconstructed, expansion, coordinate)
    if any(
        reconstructed.get(word, Fraction()) != target.get(word, Fraction())
        for word in words
    ):
        raise ArithmeticError("Hall-basis reconstruction did not reproduce BCH words")
    return coordinates


@lru_cache(maxsize=8)
def _all_hall_terms(maximum_degree: int) -> tuple[BCHHallTerm, ...]:
    maximum_degree = _validate_degree(maximum_degree)
    associative = dict(_associative_bch(maximum_degree))
    result: list[BCHHallTerm] = []
    for degree in range(1, maximum_degree + 1):
        basis_entries = tuple(entry for entry in _HALL_BASIS if entry[0] == degree)
        target = {
            word: coefficient
            for word, coefficient in associative.items()
            if len(word) == degree
        }
        coordinates = _coordinates_in_basis(
            target,
            tuple(node for _, _, node in basis_entries),
            degree,
        )
        for (_, expression, _), coefficient in zip(
            basis_entries, coordinates, strict=True
        ):
            result.append(
                BCHHallTerm(
                    degree=degree,
                    expression=expression,
                    coefficient=coefficient,
                    x_count=expression.count("X"),
                    y_count=expression.count("Y"),
                )
            )
    return tuple(result)


def bch_hall_terms(
    degree: int = 4, *, include_zero: bool = False
) -> tuple[BCHHallTerm, ...]:
    """Return exact BCH coordinates in the selected Hall-type basis."""
    terms = _all_hall_terms(_validate_degree(degree))
    if include_zero:
        return terms
    return tuple(term for term in terms if term.coefficient)


def _mobius(value: int) -> int:
    remaining = value
    prime_count = 0
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor:
            factor += 1
            continue
        remaining //= factor
        if remaining % factor == 0:
            return 0
        prime_count += 1
        while remaining % factor == 0:
            remaining //= factor
        factor += 1
    if remaining > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def _free_lie_dimension(degree: int) -> int:
    numerator: int = sum(
        _mobius(divisor) * 2 ** (degree // divisor)
        for divisor in range(1, degree + 1)
        if degree % divisor == 0
    )
    return numerator // degree


def bch_exact_certificate(degree: int = 4) -> BCHExactCertificate:
    """Certify the exact associative-log reconstruction through ``degree``."""
    maximum_degree = _validate_degree(degree)
    terms = _all_hall_terms(maximum_degree)
    dimensions = tuple(
        _free_lie_dimension(value) for value in range(1, maximum_degree + 1)
    )
    if dimensions != tuple(
        sum(term.degree == value for term in terms) for value in range(1, degree + 1)
    ):
        raise ArithmeticError("Hall basis count disagrees with the Witt dimensions")
    return BCHExactCertificate(
        maximum_degree=maximum_degree,
        free_lie_dimensions=dimensions,
        homogeneous_nonzero_hall_terms=tuple(
            sum(term.degree == value and bool(term.coefficient) for term in terms)
            for value in range(1, degree + 1)
        ),
        associative_words_checked=sum(2**value for value in range(1, degree + 1)),
        hall_basis_terms_checked=len(terms),
        reconstruction_exact=True,
    )


def bch_homogeneous_terms(
    left: Sequence[Scalar],
    right: Sequence[Scalar],
    bracket: Bracket,
    *,
    degree: int = 4,
) -> tuple[Vector, ...]:
    """Evaluate each homogeneous BCH component using a supplied Lie bracket.

    The coefficients are not written down here.  They are the exact Hall
    coordinates derived by :func:`bch_hall_terms` from ``log(exp(X) exp(Y))``,
    and each Hall expression is evaluated with ``bracket`` after substituting
    ``left`` for ``X`` and ``right`` for ``Y``.
    """
    maximum_degree = _validate_degree(degree)
    first = tuple(left)
    second = tuple(right)
    dimension = len(first)
    if len(second) != dimension:
        raise ValueError("BCH inputs must have the same dimension")

    values: dict[_LieNode, Vector] = {_X: first, _Y: second}

    def evaluate(node: _LieNode) -> Vector:
        known = values.get(node)
        if known is not None:
            return known
        if node.left is None or node.right is None:
            raise ArithmeticError("malformed internal Lie expression")
        result = tuple(bracket(evaluate(node.left), evaluate(node.right)))
        if len(result) != dimension:
            raise ValueError("the supplied bracket changed the vector dimension")
        values[node] = result
        return result

    nodes = {expression: node for _, expression, node in _HALL_BASIS}
    zero = first[0] * 0 + second[0] * 0 if dimension else Fraction()
    terms: list[Vector] = []
    for current in range(1, maximum_degree + 1):
        component: list[Scalar] = [zero for _ in range(dimension)]
        for term in _all_hall_terms(maximum_degree):
            if term.degree != current or not term.coefficient:
                continue
            for index, entry in enumerate(evaluate(nodes[term.expression])):
                component[index] += term.coefficient * entry
        terms.append(tuple(component))
    return tuple(terms)


def sum_bch_homogeneous_terms(terms: Sequence[Sequence[Scalar]]) -> Vector:
    """Sum a nonempty sequence of equal-dimensional BCH components."""
    components = tuple(tuple(term) for term in terms)
    if not components:
        raise ValueError("at least one BCH homogeneous term is required")
    dimension = len(components[0])
    if any(len(component) != dimension for component in components):
        raise ValueError("BCH homogeneous terms must have equal dimensions")
    return tuple(
        sum(component[index] for component in components) for index in range(dimension)
    )
