"""Sparse finite structure constants for arbitrary-arity multilinear maps.

The canonical coefficient key is ``(i_0, ..., i_(k-1), j)`` for a map of
arity ``k``: each ``i_r`` indexes the corresponding ordered input basis and
``j`` indexes the ordered output basis.  Thus a nullary map uses ``(j,)``.
Only declared nonzero coefficients are stored; this module never constructs a
dense tensor product of basis dimensions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import cast

from anyalgebra.core.coercions import CoercionGraph, CoercionPlan
from anyalgebra.core.domains import Domain, DomainElement
from anyalgebra.core.elements import ModuleArithmeticError, SparseElement
from anyalgebra.core.errors import AnyAlgebraError, CoercionError
from anyalgebra.core.modules import Basis, FreeModule


class MultilinearDefinitionError(AnyAlgebraError, ValueError):
    """Sparse structure-constant metadata or evaluation was malformed."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        entry_index: int | None = None,
        key: object | None = None,
    ) -> None:
        self.field = field
        self.reason = reason
        self.entry_index = entry_index
        self.key = key
        location = field if entry_index is None else f"{field}[{entry_index}]"
        super().__init__(f"invalid multilinear {location}: {reason}")


class MultilinearEvaluationError(AnyAlgebraError):
    """Exact element evaluation violated a stable public boundary."""

    def __init__(
        self,
        *,
        code: str,
        reason: str,
        argument_index: int | None = None,
        expected_parent: object | None = None,
        actual_parent: object | None = None,
    ) -> None:
        """Retain structured witnesses without rendering parent addresses."""
        self.code = code
        self.reason = reason
        self.argument_index = argument_index
        self.expected_parent = expected_parent
        self.actual_parent = actual_parent
        location = "" if argument_index is None else f" at argument {argument_index}"
        super().__init__(f"multilinear evaluation {code}{location}: {reason}")


SparseEntries = (
    Iterable[tuple[tuple[int, ...], DomainElement[object]]]
    | Mapping[tuple[int, ...], DomainElement[object]]
)


def _snapshot(values: object, *, field: str) -> tuple[object, ...]:
    """Consume an iterable or mapping-items view once with typed failure."""
    if isinstance(values, Mapping):
        values = values.items()
    if isinstance(values, str | bytes):
        raise MultilinearDefinitionError(
            field=field, reason="must be an iterable of declared entries"
        )
    try:
        return tuple(values)  # type: ignore[arg-type]
    except Exception as error:
        raise MultilinearDefinitionError(
            field=field, reason="could not be snapshotted as an iterable"
        ) from error


def _snapshot_input_bases(values: object) -> tuple[Basis, ...]:
    """Freeze ordered exact input bases, including the valid empty tuple."""
    snapshot = _snapshot(values, field="input_bases")
    bases: list[Basis] = []
    for index, basis in enumerate(snapshot):
        if type(basis) is not Basis:
            raise MultilinearDefinitionError(
                field="input_bases",
                reason="must contain exact Basis values",
                entry_index=index,
            )
        bases.append(basis)
    return tuple(bases)


def _require_domain_element(
    value: object, domain: Domain[object], *, entry_index: int
) -> DomainElement[object]:
    """Require one coefficient owned by the literal declared domain."""
    try:
        is_element = isinstance(value, DomainElement)
    except Exception as error:
        raise MultilinearDefinitionError(
            field="entries",
            reason="coefficient protocol check failed",
            entry_index=entry_index,
        ) from error
    if not is_element:
        raise MultilinearDefinitionError(
            field="entries",
            reason="coefficient must be an exact domain element",
            entry_index=entry_index,
        )
    element = cast(DomainElement[object], value)
    try:
        parent = element.parent
    except Exception as error:
        raise MultilinearDefinitionError(
            field="entries",
            reason="coefficient parent lookup failed",
            entry_index=entry_index,
        ) from error
    if parent is not domain:
        raise MultilinearDefinitionError(
            field="entries",
            reason="coefficient must have the literal coefficient parent",
            entry_index=entry_index,
        )
    return element


def _is_proven_zero(coefficient: DomainElement[object], domain: Domain[object]) -> bool:
    """Discard only a coefficient exactly equal to the domain's constructed zero."""
    try:
        zero = domain.element(0)
    except Exception:
        return False
    try:
        if not isinstance(zero, DomainElement) or zero.parent is not domain:
            return False
        result = coefficient == zero
    except Exception:
        return False
    return type(result) is bool and result is True


def _coefficients_equal(
    left: DomainElement[object], right: DomainElement[object]
) -> bool:
    """Compare coefficients without treating truthy or failed equality as proof."""
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result is True


def _validate_structure_name(name: str | None) -> str | None:
    """Validate optional display metadata before any construction input is read."""
    if name is not None and (type(name) is not str or not name or name != name.strip()):
        raise MultilinearDefinitionError(
            field="name", reason="must be None or a non-empty trimmed built-in str"
        )
    return name


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StructureConstants:
    """One immutable sparse tensor for an arbitrary finite-arity operation."""

    input_bases: tuple[Basis, ...]
    output_basis: Basis
    coefficient_domain: Domain[object]
    entries: tuple[tuple[tuple[int, ...], DomainElement[object]], ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_sparse` instead."""
        del args, kwargs
        raise MultilinearDefinitionError(
            field="constants", reason="use StructureConstants.from_sparse"
        )

    @property
    def arity(self) -> int:
        """Return the number of ordered input bases."""
        return len(self.input_bases)

    @property
    def key_shape(self) -> str:
        """Return the documented sparse-key convention without dense expansion."""
        return "(input_index_0, ..., input_index_{arity-1}, output_index)"

    @classmethod
    def from_sparse(
        cls,
        input_bases: Iterable[Basis],
        output_basis: Basis,
        entries: SparseEntries,
    ) -> StructureConstants:
        """Validate and canonically freeze sparse ``(input..., output)`` data."""
        bases = _snapshot_input_bases(input_bases)
        if type(output_basis) is not Basis:
            raise MultilinearDefinitionError(
                field="output_basis", reason="must be an exact Basis"
            )
        domain = output_basis.coefficient_domain
        for index, basis in enumerate(bases):
            if basis.coefficient_domain is not domain:
                raise MultilinearDefinitionError(
                    field="input_bases",
                    reason="every basis must have the literal same coefficient parent",
                    entry_index=index,
                )
        source = _snapshot(entries, field="entries")
        dimensions = tuple(basis.rank for basis in (*bases, output_basis))
        seen: set[tuple[int, ...]] = set()
        canonical: list[tuple[tuple[int, ...], DomainElement[object]]] = []
        for entry_index, declaration in enumerate(source):
            if type(declaration) is not tuple:
                raise MultilinearDefinitionError(
                    field="entries",
                    reason="entry must be an exact tuple",
                    entry_index=entry_index,
                )
            if len(declaration) != 2:
                raise MultilinearDefinitionError(
                    field="entries",
                    reason="entry must be a pair",
                    entry_index=entry_index,
                )
            key, raw_coefficient = declaration
            if type(key) is not tuple:
                raise MultilinearDefinitionError(
                    field="entries",
                    reason="key must be an exact tuple",
                    entry_index=entry_index,
                )
            if len(key) != len(dimensions):
                raise MultilinearDefinitionError(
                    field="entries",
                    reason="key has wrong length",
                    entry_index=entry_index,
                    key=key,
                )
            for _index, (coordinate, dimension) in enumerate(
                zip(key, dimensions, strict=True)
            ):
                if type(coordinate) is not int or coordinate < 0:
                    raise MultilinearDefinitionError(
                        field="entries",
                        reason="key entries must be exact non-negative built-in ints",
                        entry_index=entry_index,
                        key=key,
                    )
                if coordinate >= dimension:
                    raise MultilinearDefinitionError(
                        field="entries",
                        reason="key entry is outside the declared basis range",
                        entry_index=entry_index,
                        key=key,
                    )
            if key in seen:
                raise MultilinearDefinitionError(
                    field="entries",
                    reason="duplicate sparse coefficient key",
                    entry_index=entry_index,
                    key=key,
                )
            seen.add(key)
            coefficient = _require_domain_element(
                raw_coefficient, domain, entry_index=entry_index
            )
            if not _is_proven_zero(coefficient, domain):
                canonical.append((key, coefficient))
        value = object.__new__(cls)
        object.__setattr__(value, "input_bases", bases)
        object.__setattr__(value, "output_basis", output_basis)
        object.__setattr__(value, "coefficient_domain", domain)
        object.__setattr__(
            value, "entries", tuple(sorted(canonical, key=lambda item: item[0]))
        )
        return value

    def coefficients_for_basis(
        self, *input_indices: int
    ) -> tuple[tuple[int, DomainElement[object]], ...]:
        """Return sparse output coordinates for one ordered basis-input tuple."""
        if len(input_indices) != self.arity:
            raise MultilinearDefinitionError(
                field="input_indices", reason="wrong number of basis indices"
            )
        for position, (index, basis) in enumerate(
            zip(input_indices, self.input_bases, strict=True)
        ):
            if type(index) is not int or index < 0 or index >= basis.rank:
                raise MultilinearDefinitionError(
                    field="input_indices",
                    reason="basis index is outside the declared input basis range",
                    entry_index=position,
                    key=input_indices,
                )
        return tuple(
            (key[-1], coefficient)
            for key, coefficient in self.entries
            if key[:-1] == input_indices
        )

    def __eq__(self, other: object) -> bool:
        """Compare defining bases and coefficients without requiring hashability."""
        if type(other) is not StructureConstants:
            return False
        if (
            self.coefficient_domain is not other.coefficient_domain
            or self.input_bases != other.input_bases
            or self.output_basis != other.output_basis
            or len(self.entries) != len(other.entries)
        ):
            return False
        return all(
            left_key == right_key and _coefficients_equal(left, right)
            for (left_key, left), (right_key, right) in zip(
                self.entries, other.entries, strict=True
            )
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MultilinearOperation:
    """One operation view over validated sparse structure constants."""

    constants: StructureConstants
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked operation records; use the named factory."""
        del args, kwargs
        raise MultilinearDefinitionError(
            field="operation",
            reason="use MultilinearOperation.from_structure_constants",
        )

    @classmethod
    def from_structure_constants(
        cls, constants: StructureConstants
    ) -> MultilinearOperation:
        """Freeze one exact sparse constant record as an operation view."""
        if type(constants) is not StructureConstants:
            raise MultilinearDefinitionError(
                field="constants", reason="must be an exact StructureConstants"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "constants", constants)
        return value

    @property
    def arity(self) -> int:
        """Expose the declared operation arity."""
        return self.constants.arity

    def coefficients_for_basis(
        self, *input_indices: int
    ) -> tuple[tuple[int, DomainElement[object]], ...]:
        """Delegate basis-index lookup to the immutable constant tensor."""
        return self.constants.coefficients_for_basis(*input_indices)

    def __eq__(self, other: object) -> bool:
        """Compare the immutable constant tensor structurally."""
        return type(other) is MultilinearOperation and self.constants == other.constants


def _snapshot_table_cells(values: object) -> tuple[object, ...]:
    """Snapshot ordered cells without accepting a coordinate mapping as a table."""
    if isinstance(values, Mapping):
        raise MultilinearDefinitionError(
            field="cells", reason="must be an ordered iterable of table cells"
        )
    return _snapshot(values, field="cells")


def _cell_element(
    cell: object, module: FreeModule, *, cell_index: int
) -> SparseElement:
    """Normalize one exact sparse output cell into a literal-module element."""
    if type(cell) is SparseElement:
        if cell.parent is not module:
            raise MultilinearDefinitionError(
                field="cells",
                reason="SparseElement cell must have the literal module parent",
                entry_index=cell_index,
            )
        return cell
    if not isinstance(cell, Mapping):
        raise MultilinearDefinitionError(
            field="cells",
            reason="cell must be a SparseElement or exact sparse coordinate mapping",
            entry_index=cell_index,
        )
    try:
        entries = tuple(cell.items())
    except Exception as error:
        raise MultilinearDefinitionError(
            field="cells",
            reason="cell mapping could not be snapshotted",
            entry_index=cell_index,
        ) from error
    coordinates: dict[int, DomainElement[object]] = {}
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise MultilinearDefinitionError(
                field="cells",
                reason="cell mapping entry must be an exact pair",
                entry_index=cell_index,
            )
        coordinate, raw_coefficient = entry
        if type(coordinate) is not int or coordinate < 0 or coordinate >= module.rank:
            raise MultilinearDefinitionError(
                field="cells",
                reason="cell coordinate is outside the declared output basis range",
                entry_index=cell_index,
            )
        if coordinate in coordinates:
            raise MultilinearDefinitionError(
                field="cells",
                reason="cell mapping has a duplicate output coordinate",
                entry_index=cell_index,
            )
        coefficient = _require_domain_element(
            raw_coefficient, module.domain, entry_index=cell_index
        )
        coordinates[coordinate] = coefficient
    try:
        return module.element(coordinates)
    except Exception as error:
        raise MultilinearDefinitionError(
            field="cells",
            reason="cell sparse coordinates are malformed",
            entry_index=cell_index,
        ) from error


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BasisProductTable:
    """One complete ordered basis-product table for the literal one-module slice.

    Cells are in lexicographic ``itertools.product(range(rank), repeat=arity)``
    order.  The table owns exactly one operation; later versions may assemble
    multiple named operations without changing this sparse-cell contract.
    """

    module: FreeModule
    arity: int
    cells: tuple[SparseElement, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked tables; use :meth:`from_cells` instead."""
        del args, kwargs
        raise MultilinearDefinitionError(
            field="table", reason="use BasisProductTable.from_cells"
        )

    @classmethod
    def from_cells(
        cls,
        module: FreeModule,
        arity: int,
        cells: Iterable[SparseElement | Mapping[int, DomainElement[object]]],
    ) -> BasisProductTable:
        """Validate one complete ordered table without allocating input tuples."""
        if type(module) is not FreeModule:
            raise MultilinearDefinitionError(
                field="module", reason="must be an exact FreeModule"
            )
        if type(arity) is not int or arity < 0:
            raise MultilinearDefinitionError(
                field="arity", reason="must be a non-negative built-in int"
            )
        source = _snapshot_table_cells(cells)
        expected_count = module.rank**arity
        if len(source) != expected_count:
            raise MultilinearDefinitionError(
                field="cells", reason="wrong number of table cells"
            )
        normalized = tuple(
            _cell_element(cell, module, cell_index=index)
            for index, cell in enumerate(source)
        )
        value = object.__new__(cls)
        object.__setattr__(value, "module", module)
        object.__setattr__(value, "arity", arity)
        object.__setattr__(value, "cells", normalized)
        return value

    def __eq__(self, other: object) -> bool:
        """Require the literal same module before comparing canonical cells."""
        return (
            type(other) is BasisProductTable
            and self.module is other.module
            and self.arity == other.arity
            and self.cells == other.cells
        )


def _literal_basis_vector(module: FreeModule, index: int) -> SparseElement:
    """Construct one literal-module basis vector for callable sampling."""
    try:
        one = module.domain.element(1)
    except Exception as error:
        raise MultilinearDefinitionError(
            field="module", reason="could not construct the coefficient one"
        ) from error
    if not isinstance(one, DomainElement) or one.parent is not module.domain:
        raise MultilinearDefinitionError(
            field="module", reason="could not construct the literal coefficient one"
        )
    try:
        return module.element({index: one})
    except Exception as error:
        raise MultilinearDefinitionError(
            field="module", reason="could not construct a literal basis SparseElement"
        ) from error


def _callable_exception_name(error: Exception) -> str:
    """Return only safe exception type metadata, never an exception payload."""
    name = type(error).__name__
    return name if name.isidentifier() else "exception"


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DeclaredMultilinearCallable:
    """A trusted exactness declaration for one finite basis-sampled callable.

    Construction verifies metadata only.  ``from_callables`` samples every
    ordered basis tuple and compiles those values into a table; it does not
    prove that an arbitrary Python callable is multilinear.  The caller's
    ``declared_exact=True`` is therefore an explicit trusted declaration.
    """

    module: FreeModule
    arity: int
    function: Callable[..., object] = field(repr=False, compare=False)
    declared_exact: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        module: FreeModule,
        arity: int,
        function: Callable[..., object],
        *,
        declared_exact: bool = True,
    ) -> None:
        """Freeze a callable only after complete metadata preflight."""
        if type(module) is not FreeModule:
            raise MultilinearDefinitionError(
                field="module", reason="must be an exact FreeModule"
            )
        if type(arity) is not int or arity < 0:
            raise MultilinearDefinitionError(
                field="arity", reason="must be a non-negative built-in int"
            )
        if not callable(function):
            raise MultilinearDefinitionError(
                field="function", reason="must be callable"
            )
        if declared_exact is not True:
            raise MultilinearDefinitionError(
                field="declared_exact", reason="must be the literal built-in True"
            )
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "declared_exact", declared_exact)

    def __eq__(self, other: object) -> bool:
        """Treat opaque executable declarations as identity-only values."""
        return self is other

    def __repr__(self) -> str:
        """Avoid function reprs, which can expose addresses or unstable detail."""
        return (
            "DeclaredMultilinearCallable("
            f"arity={self.arity}, declared_exact={self.declared_exact})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FiniteMultilinearStructure:
    """One free module equipped with one sparse arbitrary-arity operation.

    This first slice owns a single operation on one module, matching the
    documented ``from_structure_constants(module, constants)`` route.  A
    callable route compiles a finite basis sample but trusts its declaration of
    exact multilinearity; it is not a symbolic proof of that property.
    """

    module: FreeModule
    operation: MultilinearOperation
    name: str | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked structures; use the named construction route."""
        del args, kwargs
        raise MultilinearDefinitionError(
            field="structure",
            reason="use FiniteMultilinearStructure.from_structure_constants",
        )

    @classmethod
    def from_structure_constants(
        cls,
        module: FreeModule,
        constants: StructureConstants,
        *,
        name: str | None = None,
    ) -> FiniteMultilinearStructure:
        """Attach a same-basis sparse operation to one finite free module."""
        name = _validate_structure_name(name)
        if type(module) is not FreeModule:
            raise MultilinearDefinitionError(
                field="module", reason="must be an exact FreeModule"
            )
        if type(constants) is not StructureConstants:
            raise MultilinearDefinitionError(
                field="constants", reason="must be an exact StructureConstants"
            )
        if constants.coefficient_domain is not module.domain:
            raise MultilinearDefinitionError(
                field="constants",
                reason="must have the literal module coefficient parent",
            )
        for index, basis in enumerate((*constants.input_bases, constants.output_basis)):
            if basis is not module.basis:
                raise MultilinearDefinitionError(
                    field="constants",
                    reason="every constant basis must be the literal module basis",
                    entry_index=index,
                )
        value = object.__new__(cls)
        object.__setattr__(value, "module", module)
        object.__setattr__(
            value, "operation", MultilinearOperation.from_structure_constants(constants)
        )
        object.__setattr__(value, "name", name)
        return value

    @classmethod
    def from_tables(
        cls,
        module: FreeModule,
        operations: BasisProductTable | Iterable[BasisProductTable],
        *,
        name: str | None = None,
    ) -> FiniteMultilinearStructure:
        """Compile one complete ordered basis-product table through constants.

        The public plural parameter mirrors the documented route, but this
        one-operation v0.0 slice accepts one ``BasisProductTable`` or an
        iterable containing exactly one exact table.  It does not silently
        discard additional operations or invent names for them.
        """
        name = _validate_structure_name(name)
        if type(module) is not FreeModule:
            raise MultilinearDefinitionError(
                field="module", reason="must be an exact FreeModule"
            )
        if type(operations) is BasisProductTable:
            table = operations
        else:
            if isinstance(operations, Mapping):
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exactly one BasisProductTable",
                )
            snapshot = _snapshot(operations, field="operations")
            if len(snapshot) != 1:
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exactly one BasisProductTable",
                )
            if type(snapshot[0]) is not BasisProductTable:
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exact BasisProductTable values",
                )
            table = snapshot[0]
        if table.module is not module:
            raise MultilinearDefinitionError(
                field="operations", reason="table must have the literal module parent"
            )

        constants = StructureConstants.from_sparse(
            (module.basis for _ in range(table.arity)),
            module.basis,
            (
                ((*input_indices, output_index), coefficient)
                for input_indices, cell in zip(
                    product(range(module.rank), repeat=table.arity),
                    table.cells,
                    strict=True,
                )
                for output_index, coefficient in cell.coordinates().items()
            ),
        )
        return cls.from_structure_constants(module, constants, name=name)

    @classmethod
    def from_callables(
        cls,
        module: FreeModule,
        operations: DeclaredMultilinearCallable | Iterable[DeclaredMultilinearCallable],
        *,
        name: str | None = None,
    ) -> FiniteMultilinearStructure:
        """Sample one trusted callable on the complete ordered basis product.

        This v0.0 route accepts exactly one declaration.  It deliberately
        preflights all metadata before invoking that callable, and validates
        every sampled result as a literal-parent ``SparseElement``.
        """
        name = _validate_structure_name(name)
        if type(module) is not FreeModule:
            raise MultilinearDefinitionError(
                field="module", reason="must be an exact FreeModule"
            )
        if type(operations) is DeclaredMultilinearCallable:
            declaration = operations
        else:
            if isinstance(operations, Mapping):
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exactly one DeclaredMultilinearCallable",
                )
            snapshot = _snapshot(operations, field="operations")
            if len(snapshot) != 1:
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exactly one DeclaredMultilinearCallable",
                )
            if type(snapshot[0]) is not DeclaredMultilinearCallable:
                raise MultilinearDefinitionError(
                    field="operations",
                    reason="must contain exact DeclaredMultilinearCallable values",
                )
            declaration = snapshot[0]
        if declaration.module is not module:
            raise MultilinearDefinitionError(
                field="operations",
                reason="callable must have the literal module parent",
            )

        basis_vectors = (
            tuple(_literal_basis_vector(module, index) for index in range(module.rank))
            if declaration.arity > 0
            else ()
        )
        cells: list[SparseElement] = []
        for cell_index, input_indices in enumerate(
            product(range(module.rank), repeat=declaration.arity)
        ):
            arguments = tuple(basis_vectors[index] for index in input_indices)
            raised_exception: str | None = None
            try:
                result = declaration.function(*arguments)
            except Exception as error:
                raised_exception = _callable_exception_name(error)
            if raised_exception is not None:
                raise MultilinearDefinitionError(
                    field="callable",
                    reason=f"callable raised {raised_exception}",
                    entry_index=cell_index,
                    key=input_indices,
                )
            if type(result) is not SparseElement:
                raise MultilinearDefinitionError(
                    field="callable",
                    reason="callable must return an exact SparseElement",
                    entry_index=cell_index,
                    key=input_indices,
                )
            if result.parent is not module:
                raise MultilinearDefinitionError(
                    field="callable",
                    reason="callable result must have the literal module parent",
                    entry_index=cell_index,
                    key=input_indices,
                )
            cells.append(result)
        table = BasisProductTable.from_cells(module, declaration.arity, cells)
        return cls.from_tables(module, table, name=name)

    @property
    def arity(self) -> int:
        """Expose this structure's sole operation arity."""
        return self.operation.arity

    def evaluate_basis(self, *input_indices: int) -> SparseElement:
        """Return the sparse module element for one tuple of basis indices."""
        return self.module.element(
            dict(self.operation.coefficients_for_basis(*input_indices))
        )

    def __eq__(self, other: object) -> bool:
        """Require the literal same module parent before comparing operations."""
        return (
            type(other) is FiniteMultilinearStructure
            and self.module is other.module
            and self.operation == other.operation
        )


def _evaluation_plans(
    structure: FiniteMultilinearStructure,
    elements: tuple[object, ...],
    graph: CoercionGraph | None,
) -> tuple[tuple[SparseElement, CoercionPlan | None], ...]:
    """Validate every operand and plan all explicit scalar extensions first."""
    planned: list[tuple[SparseElement, CoercionPlan | None]] = []
    target = structure.module
    for index, candidate in enumerate(elements):
        if type(candidate) is not SparseElement:
            raise MultilinearEvaluationError(
                code="element",
                reason="operand must be an exact SparseElement",
                argument_index=index,
                expected_parent=target,
            )
        element = candidate
        if element.parent is target:
            planned.append((element, None))
            continue
        if graph is None:
            raise MultilinearEvaluationError(
                code="parent",
                reason="operand requires the literal declared module parent",
                argument_index=index,
                expected_parent=target,
                actual_parent=element.parent,
            )
        if element.parent.basis.labels != target.basis.labels:
            raise MultilinearEvaluationError(
                code="basis",
                reason="scalar extension requires exactly equal ordered basis labels",
                argument_index=index,
                expected_parent=target,
                actual_parent=element.parent,
            )
        try:
            plan = graph.plan(element.parent.domain, target.domain)
        except CoercionError as error:
            raise MultilinearEvaluationError(
                code="coercion",
                reason="no unique lossless scalar-extension route",
                argument_index=index,
                expected_parent=target,
                actual_parent=element.parent,
            ) from error
        planned.append((element, plan))
    return tuple(planned)


def _apply_evaluation_plans(
    module: FreeModule,
    planned: tuple[tuple[SparseElement, CoercionPlan | None], ...],
) -> tuple[SparseElement, ...]:
    """Apply preflighted coefficient routes and reparent only by equal labels."""
    normalized: list[SparseElement] = []
    for argument_index, (element, plan) in enumerate(planned):
        if plan is None:
            normalized.append(element)
            continue
        try:
            coordinates = {
                index: plan.apply((coefficient,))[0]
                for index, coefficient in element.coordinates().items()
            }
            normalized.append(module.element(coordinates))
        except CoercionError as error:
            raise MultilinearEvaluationError(
                code="coercion",
                reason="declared scalar-extension route failed",
                argument_index=argument_index,
                expected_parent=module,
                actual_parent=element.parent,
            ) from error
    return tuple(normalized)


def evaluate_multilinear(
    structure: FiniteMultilinearStructure,
    *elements: SparseElement,
    operation: MultilinearOperation | None = None,
    graph: CoercionGraph | None = None,
) -> SparseElement:
    """Evaluate one exact finite multilinear operation on sparse elements.

    Parent and arity checks, basis compatibility, and every coercion route are
    resolved before coefficient expansion. A graph authorizes only
    coordinate-preserving scalar extension between exactly equal ordered basis
    labels; it never infers a general basis map or changes parenthesization.
    """
    if type(structure) is not FiniteMultilinearStructure:
        raise MultilinearEvaluationError(
            code="structure",
            reason="structure must be an exact FiniteMultilinearStructure",
        )
    if graph is not None and not isinstance(graph, CoercionGraph):
        raise MultilinearEvaluationError(
            code="graph", reason="graph must be a CoercionGraph or None"
        )
    if operation is not None and operation is not structure.operation:
        raise MultilinearEvaluationError(
            code="operation",
            reason="operation must be the structure's literal declared operation",
        )
    selected = structure.operation
    if len(elements) != selected.arity:
        raise MultilinearEvaluationError(
            code="arity",
            reason="number of operands must equal the declared operation arity",
        )

    planned = _evaluation_plans(structure, cast(tuple[object, ...], elements), graph)
    normalized = _apply_evaluation_plans(structure.module, planned)
    coordinate_maps = tuple(element.coordinates() for element in normalized)
    result = structure.module.zero()
    try:
        for key, coefficient in selected.constants.entries:
            input_indices = key[:-1]
            output_index = key[-1]
            if any(
                input_index not in coordinates
                for input_index, coordinates in zip(
                    input_indices, coordinate_maps, strict=True
                )
            ):
                continue
            term = structure.module.element({output_index: coefficient})
            for input_index, coordinates in zip(
                input_indices, coordinate_maps, strict=True
            ):
                term = term.scale(coordinates[input_index])
            result = result.add(term)
    except ModuleArithmeticError as error:
        raise MultilinearEvaluationError(
            code="capability",
            reason="coefficient domain lacks a required exact arithmetic capability",
        ) from error
    return result
