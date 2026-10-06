"""Explicit parent-aware adapters between finite presentation records.

This module intentionally does not infer a module, basis, operation arity, or
presentation kind from Python container shape.  One caller-owned namespace and
one literal :class:`~anyalgebra.core.modules.FreeModule` define each adapter
bundle; conversions are ordinary V51 declarations applied by V52.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
)
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import FreeModule
from anyalgebra.presentations.graph import (
    Conversion,
    ConversionGraph,
    PresentationKind,
)


_MAX_SYMBOLIC_TERMS = 512


class PresentationAdapterError(AnyAlgebraError, ValueError):
    """An explicit symbolic/coordinate/table adapter boundary was malformed."""

    def __init__(self, reason: str) -> None:
        """Keep only deterministic adapter diagnostics."""
        self.reason = reason
        super().__init__(f"invalid presentation adapter: {reason}")


def _exception_name(error: Exception) -> str:
    """Return a payload-free exception label for hostile user iterators."""
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _require_module(value: object) -> FreeModule:
    """Require one literal exact free-module parent."""
    if type(value) is not FreeModule:
        raise PresentationAdapterError("module must be an exact FreeModule")
    return value


def _require_arity(value: object) -> int:
    """Require an explicit finite operation arity."""
    if type(value) is not int or value < 0:
        raise PresentationAdapterError("arity must be a non-negative built-in int")
    return value


def _require_namespace(value: object) -> str:
    """Require a stable caller-owned namespace rather than an object address."""
    if type(value) is not str or not value or value != value.strip():
        raise PresentationAdapterError(
            "namespace must be a non-empty trimmed built-in str"
        )
    return value


def _snapshot_terms(value: object, *, maximum: int) -> tuple[object, ...]:
    """Bound an arbitrary term declaration without retaining error payloads."""
    if isinstance(value, str | bytes):
        raise PresentationAdapterError("terms must not be a raw string iterable")
    items_failure: str | None = None
    try:
        source = value.items() if isinstance(value, Mapping) else value
    except Exception as error:
        source = None
        items_failure = _exception_name(error)
    if items_failure is not None:
        raise PresentationAdapterError(f"terms could not be iterated ({items_failure})")
    iterator_failure: str | None = None
    try:
        iterator = iter(cast(Iterable[object], source))
    except Exception as error:
        iterator = None
        iterator_failure = _exception_name(error)
    if iterator_failure is not None:
        raise PresentationAdapterError(
            f"terms could not be iterated ({iterator_failure})"
        )
    assert iterator is not None
    items: list[object] = []
    iteration_failure: str | None = None
    for _ in range(maximum + 1):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            iteration_failure = _exception_name(error)
            break
        items.append(item)
    if iteration_failure is not None:
        raise PresentationAdapterError(
            f"terms could not be iterated ({iteration_failure})"
        )
    if len(items) > maximum:
        raise PresentationAdapterError(
            f"terms exceed the maximum {maximum} symbolic terms"
        )
    return tuple(items)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BasisExpression:
    """Canonical finite named-basis expression for one literal module parent."""

    module: FreeModule
    terms: tuple[tuple[str, object], ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_terms` or :meth:`from_sparse`."""
        del args, kwargs
        raise PresentationAdapterError(
            "use BasisExpression.from_terms or BasisExpression.from_sparse"
        )

    @classmethod
    def from_terms(cls, module: object, terms: object) -> BasisExpression:
        """Normalize declared named terms through the literal module constructor."""
        if cls is not BasisExpression:
            raise PresentationAdapterError("factory requires exact BasisExpression")
        parent = _require_module(module)
        declarations = _snapshot_terms(terms, maximum=_MAX_SYMBOLIC_TERMS)
        coordinates: dict[int, object] = {}
        for index, declaration in enumerate(declarations):
            if type(declaration) is not tuple or len(declaration) != 2:
                raise PresentationAdapterError(f"term {index} must be an exact pair")
            label, coefficient = declaration
            if type(label) is not str:
                raise PresentationAdapterError(
                    f"term {index} label must be an exact built-in str"
                )
            label_failure = False
            try:
                basis_index = parent.basis.index(label)
            except Exception:
                label_failure = True
                basis_index = -1
            if label_failure:
                raise PresentationAdapterError(
                    f"term {index} label is not present in the module basis"
                )
            if basis_index in coordinates:
                raise PresentationAdapterError(f"term {index} has a duplicate label")
            coordinates[basis_index] = coefficient
        construction_failure: str | None = None
        try:
            sparse = parent.element(coordinates)
        except Exception as error:
            construction_failure = _exception_name(error)
            sparse = None
        if construction_failure is not None:
            raise PresentationAdapterError(
                "terms could not be normalized by the literal module "
                f"({construction_failure})"
            )
        assert type(sparse) is SparseElement
        return cls.from_sparse(parent, sparse)

    @classmethod
    def from_sparse(cls, module: object, value: object) -> BasisExpression:
        """Reconstruct a canonical named expression from a literal-parent vector."""
        if cls is not BasisExpression:
            raise PresentationAdapterError("factory requires exact BasisExpression")
        parent = _require_module(module)
        if type(value) is not SparseElement:
            raise PresentationAdapterError(
                "coordinate value must be an exact SparseElement"
            )
        if value.parent is not parent:
            raise PresentationAdapterError(
                "coordinate value must have the literal module parent"
            )
        expression = object.__new__(cls)
        object.__setattr__(expression, "module", parent)
        object.__setattr__(
            expression,
            "terms",
            tuple(
                (parent.basis[index], coefficient)
                for index, coefficient in value.coordinates().items()
            ),
        )
        return expression

    def to_sparse(self) -> SparseElement:
        """Return the literal-parent coordinate value represented by these terms."""
        coordinates = {
            self.module.basis.index(label): coefficient
            for label, coefficient in self.terms
        }
        return self.module.element(coordinates)

    def __eq__(self, other: object) -> bool:
        """Compare canonical terms only under one literal module parent."""
        return (
            type(other) is BasisExpression
            and self.module is other.module
            and self.to_sparse() == other.to_sparse()
        )

    def __repr__(self) -> str:
        """Render safe parent-free shape metadata, never arbitrary coefficients."""
        return f"BasisExpression(term_count={len(self.terms)})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PresentationAdapterBundle:
    """One namespace-qualified finite presentation adapter for a literal module."""

    module: FreeModule
    namespace: str
    arity: int
    symbolic_kind: PresentationKind
    coordinate_kind: PresentationKind
    table_kind: PresentationKind
    structure_kind: PresentationKind
    symbolic_to_coordinate: Conversion
    table_to_structure: Conversion
    graph: ConversionGraph
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked bundles; use :meth:`for_module`."""
        del args, kwargs
        raise PresentationAdapterError("use PresentationAdapterBundle.for_module")

    @classmethod
    def for_module(
        cls, module: object, namespace: object, arity: object
    ) -> PresentationAdapterBundle:
        """Bind all four explicit kinds and two exact bidirectional conversions."""
        if cls is not PresentationAdapterBundle:
            raise PresentationAdapterError(
                "factory requires exact PresentationAdapterBundle"
            )
        parent = _require_module(module)
        stable_namespace = _require_namespace(namespace)
        operation_arity = _require_arity(arity)
        symbolic_kind = PresentationKind.from_id(f"{stable_namespace}.symbolic")
        coordinate_kind = PresentationKind.from_id(f"{stable_namespace}.coordinate")
        table_kind = PresentationKind.from_id(f"{stable_namespace}.table")
        structure_kind = PresentationKind.from_id(f"{stable_namespace}.structure")
        bundle = object.__new__(cls)
        object.__setattr__(bundle, "module", parent)
        object.__setattr__(bundle, "namespace", stable_namespace)
        object.__setattr__(bundle, "arity", operation_arity)
        object.__setattr__(bundle, "symbolic_kind", symbolic_kind)
        object.__setattr__(bundle, "coordinate_kind", coordinate_kind)
        object.__setattr__(bundle, "table_kind", table_kind)
        object.__setattr__(bundle, "structure_kind", structure_kind)
        symbolic_conversion = Conversion.from_callable(
            f"{stable_namespace}.symbolic-to-coordinate",
            symbolic_kind,
            coordinate_kind,
            bundle.coordinates_from_symbolic,
            inverse=bundle.symbolic_from_coordinates,
            guarantees=("exact", "basis-order-preserving"),
            validity_domain=f"literal-module:{stable_namespace}",
            round_trip_id=f"{stable_namespace}.symbolic-coordinate",
        )
        table_conversion = Conversion.from_callable(
            f"{stable_namespace}.table-to-structure",
            table_kind,
            structure_kind,
            bundle.structure_from_table,
            inverse=bundle.table_from_structure,
            guarantees=("exact", "basis-order-preserving"),
            validity_domain=f"literal-module:{stable_namespace};arity:{operation_arity}",
            round_trip_id=f"{stable_namespace}.table-structure",
        )
        graph = ConversionGraph.from_conversions(
            (symbolic_conversion, table_conversion)
        )
        object.__setattr__(bundle, "symbolic_to_coordinate", symbolic_conversion)
        object.__setattr__(bundle, "table_to_structure", table_conversion)
        object.__setattr__(bundle, "graph", graph)
        return bundle

    def coordinates_from_symbolic(self, value: object) -> SparseElement:
        """Convert one owned symbolic expression to its literal sparse vector."""
        if type(value) is not BasisExpression or value.module is not self.module:
            raise PresentationAdapterError(
                "symbolic value must be an exact BasisExpression of this literal module"
            )
        return value.to_sparse()

    def symbolic_from_coordinates(self, value: object) -> BasisExpression:
        """Convert one literal sparse vector to a canonical named expression."""
        return BasisExpression.from_sparse(self.module, value)

    def structure_from_table(self, value: object) -> FiniteMultilinearStructure:
        """Compile one complete owned table through existing validated constants."""
        if type(value) is not BasisProductTable:
            raise PresentationAdapterError(
                "table value must be an exact BasisProductTable"
            )
        if value.module is not self.module:
            raise PresentationAdapterError(
                "table value must have the literal module parent"
            )
        if value.arity != self.arity:
            raise PresentationAdapterError(
                "table value must have the declared adapter arity"
            )
        failure: str | None = None
        try:
            structure = FiniteMultilinearStructure.from_tables(self.module, value)
        except Exception as error:
            failure = _exception_name(error)
            structure = None
        if failure is not None:
            raise PresentationAdapterError(
                f"table could not be compiled through structure constants ({failure})"
            )
        assert type(structure) is FiniteMultilinearStructure
        return structure

    def table_from_structure(self, value: object) -> BasisProductTable:
        """Reconstruct table cells in documented lexicographic Cartesian order."""
        if type(value) is not FiniteMultilinearStructure:
            raise PresentationAdapterError(
                "structure value must be an exact FiniteMultilinearStructure"
            )
        if value.module is not self.module:
            raise PresentationAdapterError(
                "structure value must have the literal module parent"
            )
        if value.arity != self.arity:
            raise PresentationAdapterError(
                "structure value must have the declared adapter arity"
            )
        cells: list[SparseElement] = []
        evaluation_failure: str | None = None
        for input_indices in product(range(self.module.rank), repeat=self.arity):
            try:
                cell = value.evaluate_basis(*input_indices)
            except Exception as error:
                evaluation_failure = _exception_name(error)
                break
            if type(cell) is not SparseElement or cell.parent is not self.module:
                raise PresentationAdapterError(
                    "structure evaluation did not return a literal-parent SparseElement"
                )
            cells.append(cell)
        if evaluation_failure is not None:
            raise PresentationAdapterError(
                f"structure basis evaluation failed ({evaluation_failure})"
            )
        table_failure: str | None = None
        try:
            table = BasisProductTable.from_cells(self.module, self.arity, cells)
        except Exception as error:
            table_failure = _exception_name(error)
            table = None
        if table_failure is not None:
            raise PresentationAdapterError(
                f"structure cells could not form a complete table ({table_failure})"
            )
        assert type(table) is BasisProductTable
        return table

    def register_into(self, graph: object) -> ConversionGraph:
        """Persistently add this bundle's declarations to one exact existing graph."""
        if type(graph) is not ConversionGraph:
            raise PresentationAdapterError("graph must be an exact ConversionGraph")
        return graph.with_conversion(self.symbolic_to_coordinate).with_conversion(
            self.table_to_structure
        )

    def __eq__(self, other: object) -> bool:
        """Keep callable-bearing bundles identity-only."""
        return self is other

    def __repr__(self) -> str:
        """Render only stable namespace/arity metadata."""
        return (
            f"PresentationAdapterBundle(namespace={self.namespace!r}, "
            f"arity={self.arity})"
        )
