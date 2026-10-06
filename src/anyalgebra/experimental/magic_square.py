"""Catalogues and exact controls for the 2-by-2 and 3-by-3 magic squares.

The module keeps three objects separate:

* the raw carrier ``A tensor J_n(B)``;
* the Lie algebra constructed from that data; and
* a connected Lie-group name, understood only up to finite cover unless a
  particular global form is stated.

For ``n=2`` the Lie algebra is the orthogonal square
``so(A direct_sum B)``.  For ``n=3`` the catalogue records the real
Freudenthal--Tits magic square.  This is an experimental research API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from math import comb
from types import MappingProxyType
from typing import TypeAlias


IntegerMatrix: TypeAlias = tuple[tuple[int, ...], ...]
Signature: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CompositionAlgebraSpec:
    """One real unital composition algebra and its norm signature."""

    key: str
    symbol: str
    base: str
    dimension: int
    positive: int
    negative: int
    split: bool

    @property
    def signature(self) -> Signature:
        return self.positive, self.negative


@dataclass(frozen=True, slots=True)
class MagicSquareEntry:
    """One catalogued Lie algebra obtained from ``A tensor J_n(B)`` data.

    ``group`` is a conventional source label, not a choice of center or
    fundamental group.  ``group_scope`` records that boundary explicitly.
    """

    matrix_size: int
    left: CompositionAlgebraSpec
    right: CompositionAlgebraSpec
    carrier_dimension: int
    lie_algebra_dimension: int
    lie_algebra: str
    group: str
    group_scope: str
    decomposition: tuple[int, int, int]
    signature: Signature | None
    construction: str
    source: str


@dataclass(frozen=True, slots=True)
class OrthogonalCertificate:
    """Finite exact receipt for the standard ``so(p,q)`` generator basis."""

    signature: Signature
    generator_count: int
    metric_skew_generators: int
    commutators_checked: int
    closed: bool


@dataclass(frozen=True, slots=True)
class _N3Form:
    algebra: str
    group: str
    dimension: int


_ALGEBRAS = MappingProxyType(
    {
        "R": CompositionAlgebraSpec("R", "R", "R", 1, 1, 0, False),
        "C": CompositionAlgebraSpec("C", "C", "C", 2, 2, 0, False),
        "Cs": CompositionAlgebraSpec("Cs", "C_s", "C", 2, 1, 1, True),
        "H": CompositionAlgebraSpec("H", "H", "H", 4, 4, 0, False),
        "Hs": CompositionAlgebraSpec("Hs", "H_s", "H", 4, 2, 2, True),
        "O": CompositionAlgebraSpec("O", "O", "O", 8, 8, 0, False),
        "Os": CompositionAlgebraSpec("Os", "O_s", "O", 8, 4, 4, True),
    }
)

_BASE_ORDER = {"R": 0, "C": 1, "H": 2, "O": 3}
_DERIVATION_DIMENSION = {"R": 0, "C": 0, "H": 3, "O": 14}
_J3_DERIVATION_DIMENSION = {"R": 3, "C": 8, "H": 21, "O": 52}
_TRIALITY_DIMENSION = {"R": 0, "C": 2, "H": 9, "O": 28}


def _base_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right), key=_BASE_ORDER.__getitem__))  # type: ignore[return-value]


_COMPACT_N3 = MappingProxyType(
    {
        ("R", "R"): _N3Form("so(3)", "SO(3)", 3),
        ("R", "C"): _N3Form("su(3)", "SU(3)", 8),
        ("R", "H"): _N3Form("sp(3)", "Sp(3)", 21),
        ("R", "O"): _N3Form("f4(-52)", "F4(-52)", 52),
        ("C", "C"): _N3Form("su(3)+su(3)", "SU(3)xSU(3)", 16),
        ("C", "H"): _N3Form("su(6)", "SU(6)", 35),
        ("C", "O"): _N3Form("e6(-78)", "E6(-78)", 78),
        ("H", "H"): _N3Form("so(12)", "Spin(12)", 66),
        ("H", "O"): _N3Form("e7(-133)", "E7(-133)", 133),
        ("O", "O"): _N3Form("e8(-248)", "E8(-248)", 248),
    }
)

# One split algebra and one division algebra.  These are the entries of the
# Gunaydin--Sierra--Townsend single-split square.
_HALF_SPLIT_N3 = MappingProxyType(
    {
        ("C", "R"): _N3Form("sl(3,R)", "SL(3,R)", 8),
        ("C", "C"): _N3Form("sl(3,C)_R", "SL(3,C)", 16),
        ("C", "H"): _N3Form("su*(6)", "SU*(6)", 35),
        ("C", "O"): _N3Form("e6(-26)", "E6(-26)", 78),
        ("H", "R"): _N3Form("sp(6,R)", "Sp(6,R)", 21),
        ("H", "C"): _N3Form("su(3,3)", "SU(3,3)", 35),
        ("H", "H"): _N3Form("so*(12)", "Spin*(12)", 66),
        ("H", "O"): _N3Form("e7(-25)", "E7(-25)", 133),
        ("O", "R"): _N3Form("f4(4)", "F4(4)", 52),
        ("O", "C"): _N3Form("e6(2)", "E6(2)", 78),
        ("O", "H"): _N3Form("e7(-5)", "E7(-5)", 133),
        ("O", "O"): _N3Form("e8(-24)", "E8(-24)", 248),
    }
)

_DOUBLE_SPLIT_N3 = MappingProxyType(
    {
        ("C", "C"): _N3Form("sl(3,R)+sl(3,R)", "SL(3,R)xSL(3,R)", 16),
        ("C", "H"): _N3Form("sl(6,R)", "SL(6,R)", 35),
        ("C", "O"): _N3Form("e6(6)", "E6(6)", 78),
        ("H", "H"): _N3Form("so(6,6)", "Spin(6,6)", 66),
        ("H", "O"): _N3Form("e7(7)", "E7(7)", 133),
        ("O", "O"): _N3Form("e8(8)", "E8(8)", 248),
    }
)


def composition_algebra_keys() -> tuple[str, ...]:
    """Return the deterministic catalogue order."""
    return tuple(_ALGEBRAS)


def composition_algebra(key: str) -> CompositionAlgebraSpec:
    """Look up a division or split composition algebra by short key."""
    try:
        return _ALGEBRAS[key]
    except (KeyError, TypeError) as error:
        options = ", ".join(composition_algebra_keys())
        raise ValueError(
            f"unknown composition algebra {key!r}; expected one of {options}"
        ) from error


def triality_dimension(algebra: str | CompositionAlgebraSpec) -> int:
    """Return the dimension of the triality algebra of a composition algebra.

    Division and split forms have the same dimension. Their real forms and
    invariant-form signatures differ.
    """

    spec = composition_algebra(algebra) if isinstance(algebra, str) else algebra
    if not isinstance(spec, CompositionAlgebraSpec):
        raise TypeError("algebra must be a catalogue key or CompositionAlgebraSpec")
    return _TRIALITY_DIMENSION[spec.base]


def n3_triality_decomposition(left: str, right: str) -> tuple[int, int, int]:
    """Return ``dim Tri(A), dim Tri(B), 3 dim(A tensor B)``.

    This is the symmetric Barton--Sudbery decomposition of the degree-three
    magic-square Lie algebra. It is distinct from the asymmetric Tits
    decomposition stored on :class:`MagicSquareEntry`.
    """

    a = composition_algebra(left)
    b = composition_algebra(right)
    return (
        triality_dimension(a),
        triality_dimension(b),
        3 * a.dimension * b.dimension,
    )


def jordan_dimension(matrix_size: int, algebra: str | CompositionAlgebraSpec) -> int:
    """Return ``dim H_n(B)`` for ``n`` equal to two or three."""
    if type(matrix_size) is not int or matrix_size not in {2, 3}:
        raise ValueError("matrix_size must be 2 or 3")
    spec = composition_algebra(algebra) if isinstance(algebra, str) else algebra
    if not isinstance(spec, CompositionAlgebraSpec):
        raise TypeError("algebra must be a catalogue key or CompositionAlgebraSpec")
    return matrix_size + comb(matrix_size, 2) * spec.dimension


def _orthogonal_name(signature: Signature, *, group: bool) -> str:
    positive, negative = signature
    stem = "Spin" if group else "so"
    if negative == 0:
        return f"{stem}({positive})"
    return f"{stem}({positive},{negative})"


def _n2_source(a: CompositionAlgebraSpec, b: CompositionAlgebraSpec) -> str:
    """Return the published treatment closest to one orthogonal-square entry.

    Dray, Huerta, and Kincaid tabulate the cases with exactly one split
    factor.  For the other entries the label is the orthogonal algebra of the
    direct sum of the two norm forms, whose signature is computed here.
    """
    if a.split != b.split:
        return "Dray--Huerta--Kincaid 2014, 2-by-2 magic square"
    return (
        "Barton--Sudbery 2003, so(A + B) form of the 2-by-2 square; "
        "signature computed from the two norm forms"
    )


@cache
def n2_magic_square_entry(left: str, right: str) -> MagicSquareEntry:
    """Return the orthogonal-square entry for ``A tensor J_2(B)`` data."""
    a = composition_algebra(left)
    b = composition_algebra(right)
    signature = (a.positive + b.positive, a.negative + b.negative)
    dimension = a.dimension + b.dimension
    # so(Im A) + so(J_2(B)_0) + Im A tensor J_2(B)_0
    decomposition = (
        comb(a.dimension - 1, 2),
        comb(b.dimension + 1, 2),
        (a.dimension - 1) * (b.dimension + 1),
    )
    return MagicSquareEntry(
        matrix_size=2,
        left=a,
        right=b,
        carrier_dimension=a.dimension * jordan_dimension(2, b),
        lie_algebra_dimension=comb(dimension, 2),
        lie_algebra=_orthogonal_name(signature, group=False),
        group=_orthogonal_name(signature, group=True),
        group_scope="connected double cover of SO_0(p,q)",
        decomposition=decomposition,
        signature=signature,
        construction="so(Im(A)) + so(J2(B)_0) + Im(A) tensor J2(B)_0",
        source=_n2_source(a, b),
    )


def _n3_form(a: CompositionAlgebraSpec, b: CompositionAlgebraSpec) -> _N3Form:
    pair = _base_pair(a.base, b.base)
    if not a.split and not b.split:
        return _COMPACT_N3[pair]
    if a.split and b.split:
        return _DOUBLE_SPLIT_N3[pair]
    split = a if a.split else b
    division = b if a.split else a
    return _HALF_SPLIT_N3[(split.base, division.base)]


def _n3_source(a: CompositionAlgebraSpec, b: CompositionAlgebraSpec) -> str:
    """Return the published table that fixes the real-form label."""
    if not a.split and not b.split:
        table = 1
    elif a.split and b.split:
        table = 3
    else:
        table = 2
    return f"Cacciatori--Cerchiai--Marrani 2012, Table {table}"


@cache
def n3_magic_square_entry(left: str, right: str) -> MagicSquareEntry:
    """Return one real Freudenthal--Tits 3-by-3 magic-square entry."""
    a = composition_algebra(left)
    b = composition_algebra(right)
    trace_free_jordan = jordan_dimension(3, b) - 1
    decomposition = (
        _DERIVATION_DIMENSION[a.base],
        _J3_DERIVATION_DIMENSION[b.base],
        (a.dimension - 1) * trace_free_jordan,
    )
    form = _n3_form(a, b)
    dimension = sum(decomposition)
    if dimension != form.dimension:
        raise AssertionError(
            f"catalogue dimension mismatch for {left} tensor J3({right})"
        )
    return MagicSquareEntry(
        matrix_size=3,
        left=a,
        right=b,
        carrier_dimension=a.dimension * jordan_dimension(3, b),
        lie_algebra_dimension=dimension,
        lie_algebra=form.algebra,
        group=form.group,
        group_scope="connected global form unspecified",
        decomposition=decomposition,
        signature=None,
        construction="Der(A) + Der(J3(B)) + Im(A) tensor J3(B)_0",
        source=_n3_source(a, b),
    )


def magic_square(
    matrix_size: int, keys: Sequence[str] | None = None
) -> tuple[tuple[MagicSquareEntry, ...], ...]:
    """Return a square over all requested composition-algebra keys."""
    if type(matrix_size) is not int or matrix_size not in {2, 3}:
        raise ValueError("matrix_size must be 2 or 3")
    selected = composition_algebra_keys() if keys is None else tuple(keys)
    for key in selected:
        composition_algebra(key)
    entry = n2_magic_square_entry if matrix_size == 2 else n3_magic_square_entry
    return tuple(tuple(entry(left, right) for right in selected) for left in selected)


def _metric_signs(signature: Signature) -> tuple[int, ...]:
    positive, negative = signature
    return (1,) * positive + (-1,) * negative


@cache
def n2_orthogonal_basis(left: str, right: str) -> tuple[IntegerMatrix, ...]:
    """Return the standard exact plane-generator basis of ``so(p,q)``."""
    entry = n2_magic_square_entry(left, right)
    if entry.signature is None:
        raise AssertionError("an n=2 entry must carry an orthogonal signature")
    signs = _metric_signs(entry.signature)
    dimension = len(signs)
    matrices: list[IntegerMatrix] = []
    for first in range(dimension):
        for second in range(first + 1, dimension):
            rows = [[0 for _ in range(dimension)] for _ in range(dimension)]
            rows[first][second] = 1
            rows[second][first] = -signs[first] * signs[second]
            matrices.append(tuple(tuple(row) for row in rows))
    return tuple(matrices)


def _sparse_plane(
    first: int, second: int, signs: tuple[int, ...]
) -> dict[tuple[int, int], int]:
    return {
        (first, second): 1,
        (second, first): -signs[first] * signs[second],
    }


def _sparse_product(
    left: dict[tuple[int, int], int], right: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for (row, middle), left_value in left.items():
        for (other_middle, column), right_value in right.items():
            if middle != other_middle:
                continue
            key = row, column
            result[key] = result.get(key, 0) + left_value * right_value
    return {key: value for key, value in result.items() if value}


def _sparse_commutator(
    left: dict[tuple[int, int], int], right: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int]:
    lr = _sparse_product(left, right)
    rl = _sparse_product(right, left)
    keys = lr.keys() | rl.keys()
    return {
        key: lr.get(key, 0) - rl.get(key, 0)
        for key in keys
        if lr.get(key, 0) != rl.get(key, 0)
    }


@cache
def n2_orthogonal_certificate(left: str, right: str) -> OrthogonalCertificate:
    """Check metric skewness and all unordered basis commutators exactly."""
    entry = n2_magic_square_entry(left, right)
    if entry.signature is None:
        raise AssertionError("an n=2 entry must carry an orthogonal signature")
    signs = _metric_signs(entry.signature)
    pairs = tuple(
        (first, second)
        for first in range(len(signs))
        for second in range(first + 1, len(signs))
    )
    sparse_basis = tuple(_sparse_plane(i, j, signs) for i, j in pairs)

    metric_skew = 0
    for generator in sparse_basis:
        if all(
            signs[row] * generator.get((row, column), 0)
            + signs[column] * generator.get((column, row), 0)
            == 0
            for row in range(len(signs))
            for column in range(len(signs))
        ):
            metric_skew += 1

    checked = 0
    closed = True
    for left_index, left_generator in enumerate(sparse_basis):
        for right_generator in sparse_basis[left_index:]:
            bracket = _sparse_commutator(left_generator, right_generator)
            coordinates = {
                pair: bracket.get(pair, 0) for pair in pairs if bracket.get(pair, 0)
            }
            reconstructed: dict[tuple[int, int], int] = {}
            for (first, second), coefficient in coordinates.items():
                for key, value in _sparse_plane(first, second, signs).items():
                    reconstructed[key] = reconstructed.get(key, 0) + coefficient * value
            reconstructed = {
                key: value for key, value in reconstructed.items() if value
            }
            if reconstructed != bracket:
                closed = False
            checked += 1

    return OrthogonalCertificate(
        signature=entry.signature,
        generator_count=len(pairs),
        metric_skew_generators=metric_skew,
        commutators_checked=checked,
        closed=closed,
    )


def render_magic_square_markdown(
    matrix_size: int, keys: Sequence[str] | None = None
) -> str:
    """Render Lie-algebra labels as a compact Markdown table."""
    selected = composition_algebra_keys() if keys is None else tuple(keys)
    table = magic_square(matrix_size, selected)
    header = "| A \\ B | " + " | ".join(selected) + " |"
    separator = "|" + "---|" * (len(selected) + 1)
    rows = [header, separator]
    for key, entries in zip(selected, table, strict=True):
        rows.append(
            "| " + key + " | " + " | ".join(item.lie_algebra for item in entries) + " |"
        )
    return "\n".join(rows)
