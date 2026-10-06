"""Named generic finite-table fixtures for four composition conventions.

These functions load exact, finite basis-product tables through the neutral
``Basis``/``FreeModule``/``BasisProductTable``/``FiniteMultilinearStructure``
constructors.  They are test fixtures, not specialized composition-algebra
classes, and do not certify alternativity, norm properties, or any scientific
claim beyond the explicitly recorded multiplication convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
)
from anyalgebra.core.domains import ZZ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis, FreeModule


class CompositionFixtureError(AnyAlgebraError, ValueError):
    """A requested named generic composition fixture was unavailable."""


SignedProduct = tuple[int, int]
ProductTable = tuple[tuple[SignedProduct, ...], ...]


@dataclass(frozen=True, slots=True)
class _CompositionFixtureMetadata:
    """Immutable convention data for one generic finite-table fixture."""

    identifier: str
    basis_labels: tuple[str, ...]
    product_table: ProductTable
    source_anchors: tuple[str, ...]


QUATERNION_TABLE: ProductTable = (
    ((1, 0), (1, 1), (1, 2), (1, 3)),
    ((1, 1), (-1, 0), (1, 3), (-1, 2)),
    ((1, 2), (-1, 3), (-1, 0), (1, 1)),
    ((1, 3), (1, 2), (-1, 1), (-1, 0)),
)

SPLIT_QUATERNION_TABLE: ProductTable = (
    ((1, 0), (1, 1), (1, 2), (1, 3)),
    ((1, 1), (-1, 0), (1, 3), (-1, 2)),
    ((1, 2), (-1, 3), (1, 0), (-1, 1)),
    ((1, 3), (1, 2), (1, 1), (1, 0)),
)

# Exact positive triples from the `algmul.O.v1` convention manifest.  Each
# triple (a, b, c) declares a*b=c, b*c=a, and c*a=b; reversals negate output.
OCTONION_POSITIVE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)

# Exact row-major table transcribed from the normative `algmul.Os.v1` fixture.
# Each pair is ``(coefficient, output_basis_index)`` in basis
# ``(1, e1, e2, e3, e4, e5, e6, e7)``.
SPLIT_OCTONION_TABLE: ProductTable = (
    ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)),
    ((1, 1), (-1, 0), (1, 3), (-1, 2), (-1, 5), (1, 4), (-1, 7), (1, 6)),
    ((1, 2), (-1, 3), (-1, 0), (1, 1), (-1, 6), (1, 7), (1, 4), (-1, 5)),
    ((1, 3), (1, 2), (-1, 1), (-1, 0), (-1, 7), (-1, 6), (1, 5), (1, 4)),
    ((1, 4), (1, 5), (1, 6), (1, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
    ((1, 5), (-1, 4), (-1, 7), (1, 6), (-1, 1), (1, 0), (1, 3), (-1, 2)),
    ((1, 6), (1, 7), (-1, 4), (-1, 5), (-1, 2), (-1, 3), (1, 0), (1, 1)),
    ((1, 7), (-1, 6), (1, 5), (-1, 4), (-1, 3), (1, 2), (-1, 1), (1, 0)),
)


def _fano_product(left: int, right: int) -> SignedProduct:
    """Look up one nonunit, distinct octonion generator product exactly."""
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
    raise CompositionFixtureError("octonion Fano triples omit an imaginary pair")


def _octonion_table() -> ProductTable:
    """Construct the ordinary-octonion table from the auditable Fano data."""
    rows: list[tuple[SignedProduct, ...]] = []
    for left in range(8):
        row: list[SignedProduct] = []
        for right in range(8):
            if left == 0:
                row.append((1, right))
            elif right == 0:
                row.append((1, left))
            elif left == right:
                row.append((-1, 0))
            else:
                row.append(_fano_product(left, right))
        rows.append(tuple(row))
    return tuple(rows)


_FIXTURES: tuple[_CompositionFixtureMetadata, ...] = (
    _CompositionFixtureMetadata(
        identifier="algmul.H.v1",
        basis_labels=("1", "i", "j", "k"),
        product_table=QUATERNION_TABLE,
        source_anchors=("docs/conventions/conventions-v0.0.md#4.2-quaternions",),
    ),
    _CompositionFixtureMetadata(
        identifier="algmul.Hs.v1",
        basis_labels=("1", "i", "j", "k"),
        product_table=SPLIT_QUATERNION_TABLE,
        source_anchors=("docs/conventions/conventions-v0.0.md#4.3-split-quaternions",),
    ),
    _CompositionFixtureMetadata(
        identifier="algmul.O.v1",
        basis_labels=("1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"),
        product_table=_octonion_table(),
        source_anchors=("docs/conventions/conventions-v0.0.md#4.4-octonions",),
    ),
    _CompositionFixtureMetadata(
        identifier="algmul.Os.v1",
        basis_labels=("1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"),
        product_table=SPLIT_OCTONION_TABLE,
        source_anchors=(
            "docs/conventions/conventions-v0.0.md#4.5-split-octonions",
            "docs/legacy/generated/algmul-evaluated-surface.json",
        ),
    ),
)


def _structure_from_metadata(
    metadata: _CompositionFixtureMetadata,
) -> FiniteMultilinearStructure:
    """Build one table only through the current generic finite APIs."""
    domain = ZZ()
    module = FreeModule(domain, Basis(metadata.basis_labels, coefficient_domain=domain))
    cells = tuple(
        module.element({output: domain.element(coefficient)})
        for row in metadata.product_table
        for coefficient, output in row
    )
    basis_table = BasisProductTable.from_cells(module, 2, cells)
    return FiniteMultilinearStructure.from_tables(
        module, basis_table, name=metadata.identifier
    )


def _fixture_metadata(identifier: str) -> _CompositionFixtureMetadata:
    """Select one exact immutable fixture record without mutable registry state."""
    if type(identifier) is not str:
        raise CompositionFixtureError(
            "fixture identifier must be an exact built-in str"
        )
    for metadata in _FIXTURES:
        if metadata.identifier == identifier:
            return metadata
    raise CompositionFixtureError("fixture identifier is not registered")


def quaternion_fixture() -> FiniteMultilinearStructure:
    """Load the generic `algmul.H.v1` quaternion multiplication fixture."""
    return _structure_from_metadata(_fixture_metadata("algmul.H.v1"))


def split_quaternion_fixture() -> FiniteMultilinearStructure:
    """Load the generic `algmul.Hs.v1` split-quaternion table fixture."""
    return _structure_from_metadata(_fixture_metadata("algmul.Hs.v1"))


def octonion_fixture() -> FiniteMultilinearStructure:
    """Load the generic `algmul.O.v1` Fano-oriented octonion fixture."""
    return _structure_from_metadata(_fixture_metadata("algmul.O.v1"))


def split_octonion_fixture() -> FiniteMultilinearStructure:
    """Load the generic literal-table `algmul.Os.v1` split-octonion fixture."""
    return _structure_from_metadata(_fixture_metadata("algmul.Os.v1"))


def composition_fixture(identifier: str) -> FiniteMultilinearStructure:
    """Load one explicitly named generic convention fixture, never an alias."""
    return _structure_from_metadata(_fixture_metadata(identifier))
