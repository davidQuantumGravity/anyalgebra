"""Immutable finite total operations backed by complete tuple tables.

``Operation.from_table`` accepts one neutral, ordered representation for both
sources: ``carriers`` is an iterable of ``(Sort, FiniteCarrier)`` bindings and
``table`` is an iterable of ``(input_tuple, output)`` entries.  A mapping is
accepted as a convenience source for either input, but is never required: the
iterable forms support unhashable carrier items and table values.  Both sources
are snapshotted exactly once.  Every table input is an exact tuple, including
``()`` for a nullary operation.

``Operation`` deliberately remains total: missing cells are malformed
construction input.  ``PartialOperation`` is the separate table route for
mathematical undefinedness, represented without adjoining a bottom value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import (
    CarrierMemberNotFoundError,
    FiniteCarrier,
    Sort,
)
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol


class OperationDefinitionError(AnyAlgebraError, ValueError):
    """An operation table or carrier declaration was malformed.

    Locations are retained as declared indices or canonical carrier-index
    tuples.  Error messages intentionally do not render arbitrary caller
    values, whose ``repr`` may be unstable or have side effects.
    """

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        binding_index: int | None = None,
        entry_index: int | None = None,
        conflicting_binding_index: int | None = None,
        conflicting_entry_index: int | None = None,
        argument_index: int | None = None,
        index_tuple: tuple[int, ...] | None = None,
        expected_sort: Sort | None = None,
    ) -> None:
        """Store deterministic definition diagnostics without caller reprs."""
        self.field = field
        self.reason = reason
        self.binding_index = binding_index
        self.entry_index = entry_index
        self.conflicting_binding_index = conflicting_binding_index
        self.conflicting_entry_index = conflicting_entry_index
        self.argument_index = argument_index
        self.index_tuple = index_tuple
        self.expected_sort = expected_sort
        location = field
        if binding_index is not None:
            location = f"{location} binding[{binding_index}]"
        if entry_index is not None:
            location = f"{location} entry[{entry_index}]"
        if argument_index is not None:
            location = f"{location} argument[{argument_index}]"
        if index_tuple is not None:
            location = f"{location} index tuple {index_tuple}"
        detail = f"invalid {self.operation_kind} operation {location}: {reason}"
        if conflicting_binding_index is not None:
            detail = f"{detail} (conflicts with binding {conflicting_binding_index})"
        if conflicting_entry_index is not None:
            detail = f"{detail} (conflicts with entry {conflicting_entry_index})"
        super().__init__(detail)

    operation_kind = "total"


class PartialOperationDefinitionError(OperationDefinitionError):
    """A partial-operation table or marker declaration was malformed."""

    operation_kind = "partial"


class OperationApplicationError(AnyAlgebraError, ValueError):
    """A call did not satisfy an already-defined operation boundary."""

    def __init__(
        self,
        symbol: OperationSymbol,
        *,
        reason: str,
        argument_index: int | None = None,
        expected_arity: int | None = None,
        actual_arity: int | None = None,
        expected_sort: Sort | None = None,
    ) -> None:
        """Keep exact operation metadata and stable application diagnostics."""
        self.symbol = symbol
        self.reason = reason
        self.argument_index = argument_index
        self.expected_arity = expected_arity
        self.actual_arity = actual_arity
        self.expected_sort = expected_sort
        location = "arguments"
        if argument_index is not None:
            location = f"argument[{argument_index}]"
        super().__init__(
            f"invalid application of {self.operation_kind} operation "
            f"{symbol.name!r} {location}: "
            f"{reason}"
        )

    operation_kind = "total"


class PartialOperationApplicationError(OperationApplicationError):
    """A call did not satisfy an already-defined partial operation boundary."""

    operation_kind = "partial"


CarrierBindings = Iterable[tuple[Sort, FiniteCarrier]] | Mapping[Sort, FiniteCarrier]
TableEntries = (
    Iterable[tuple[tuple[object, ...], object]] | Mapping[tuple[object, ...], object]
)


def _require_callable(
    function: object,
    *,
    error_type: type[OperationDefinitionError] = OperationDefinitionError,
) -> Callable[..., object]:
    """Require a callable without probing or executing it during construction."""
    if not callable(function):
        raise error_type(field="function", reason="must be callable")
    return cast(Callable[..., object], function)


def _exception_label(error: Exception) -> str:
    """Return a safe stable exception type label without rendering the error."""
    name = type(error).__name__
    if (
        type(name) is str
        and bool(name)
        and name == name.strip()
        and name.isidentifier()
    ):
        return name
    return "exception"


def _snapshot_source(
    values: object,
    *,
    field: str,
    error_type: type[OperationDefinitionError] = OperationDefinitionError,
) -> tuple[object, ...]:
    """Consume an iterable or mapping-items view once with a typed failure."""
    if isinstance(values, Mapping):
        values = values.items()
    if isinstance(values, str | bytes):
        raise error_type(field=field, reason="must be an iterable of declared entries")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except (TypeError, RuntimeError) as error:
        raise error_type(
            field=field, reason="could not be snapshotted as an iterable"
        ) from error


def _require_pair(
    value: object,
    *,
    field: str,
    binding_index: int | None = None,
    entry_index: int | None = None,
    error_type: type[OperationDefinitionError] = OperationDefinitionError,
) -> tuple[object, object]:
    """Validate a concrete two-item declaration without consuming it twice."""
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise error_type(
            field=field,
            reason="entry must be a pair",
            binding_index=binding_index,
            entry_index=entry_index,
        )
    return value[0], value[1]


def _required_sorts(symbol: OperationSymbol) -> tuple[Sort, ...]:
    """Return distinct operation sorts in their first declared occurrence."""
    required: list[Sort] = []
    for sort in (*symbol.inputs, symbol.output):
        if sort not in required:
            required.append(sort)
    return tuple(required)


def _snapshot_carrier_bindings(
    values: object,
    symbol: OperationSymbol,
    *,
    error_type: type[OperationDefinitionError] = OperationDefinitionError,
) -> tuple[tuple[Sort, FiniteCarrier], ...]:
    """Validate one complete, exact finite-carrier binding per required sort."""
    snapshot = _snapshot_source(values, field="carriers", error_type=error_type)
    bindings: list[tuple[Sort, FiniteCarrier]] = []
    for index, declaration in enumerate(snapshot):
        raw_sort, raw_carrier = _require_pair(
            declaration,
            field="carriers",
            binding_index=index,
            error_type=error_type,
        )
        if type(raw_sort) is not Sort:
            raise error_type(
                field="carriers",
                reason="binding must contain an exact Sort",
                binding_index=index,
            )
        if type(raw_carrier) is not FiniteCarrier:
            raise error_type(
                field="carriers",
                reason="binding must contain an exact FiniteCarrier",
                binding_index=index,
            )
        if raw_carrier.sort != raw_sort:
            raise error_type(
                field="carriers",
                reason="carrier sort does not match binding sort",
                binding_index=index,
                expected_sort=raw_sort,
            )
        for prior_index, (prior_sort, _) in enumerate(bindings):
            if prior_sort == raw_sort:
                raise error_type(
                    field="carriers",
                    reason="duplicate carrier binding",
                    binding_index=index,
                    conflicting_binding_index=prior_index,
                    expected_sort=raw_sort,
                )
        bindings.append((raw_sort, raw_carrier))

    required = _required_sorts(symbol)
    for sort in required:
        if not any(bound_sort == sort for bound_sort, _ in bindings):
            raise error_type(
                field="carriers",
                reason="missing required carrier binding",
                expected_sort=sort,
            )
    return tuple(bindings)


def _carrier_for_sort(
    bindings: tuple[tuple[Sort, FiniteCarrier], ...], sort: Sort
) -> FiniteCarrier:
    """Find the already-validated literal carrier bound to one sort."""
    for bound_sort, carrier in bindings:
        if bound_sort == sort:
            return carrier
    raise AssertionError("validated operation is missing a required carrier")


def _canonical_table_member_index(
    carrier: FiniteCarrier,
    value: object,
    *,
    entry_index: int,
    argument_index: int | None,
    error_type: type[OperationDefinitionError] = OperationDefinitionError,
) -> int:
    """Canonicalize one table value without requiring it to hash."""
    try:
        return carrier.index(value)
    except CarrierMemberNotFoundError as error:
        reason = "input is not a member of its declared carrier"
        if argument_index is None:
            reason = "output is not a member of its declared carrier"
            if error.reason != "item is not a member":
                reason = "output carrier equality comparison failed"
        elif error.reason != "item is not a member":
            reason = "input carrier equality comparison failed"
        raise error_type(
            field="table",
            reason=reason,
            entry_index=entry_index,
            argument_index=argument_index,
            expected_sort=carrier.sort,
        ) from error


@dataclass(frozen=True, slots=True, init=False)
class Operation:
    """A finite total operation with a complete immutable canonical table.

    ``carrier_bindings`` retains the literal ``FiniteCarrier`` instances in
    supplied binding order, including structurally valid bindings unused by
    this operation.  ``table`` stores every Cartesian cell in
    lexicographic carrier-index order, so runtime lookup has no dependency on
    hashability of source values.  The class is deliberately unhashable because
    valid carrier items and output payloads may be unhashable.
    """

    symbol: OperationSymbol
    carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...]
    input_carriers: tuple[FiniteCarrier, ...]
    output_carrier: FiniteCarrier
    table: tuple[tuple[tuple[int, ...], object], ...]
    domain: tuple[FiniteCarrier, ...]
    totality: str
    bounds: object | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_table` instead."""
        del args, kwargs
        raise OperationDefinitionError(
            field="operation", reason="use Operation.from_table for construction"
        )

    @classmethod
    def _from_validated(
        cls,
        symbol: OperationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        table: tuple[tuple[tuple[int, ...], object], ...],
    ) -> Operation:
        """Allocate an operation after :meth:`from_table` proved its invariants."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        output_carrier = _carrier_for_sort(carrier_bindings, symbol.output)
        operation = object.__new__(cls)
        object.__setattr__(operation, "symbol", symbol)
        object.__setattr__(operation, "carrier_bindings", carrier_bindings)
        object.__setattr__(operation, "input_carriers", input_carriers)
        object.__setattr__(operation, "output_carrier", output_carrier)
        object.__setattr__(operation, "table", table)
        object.__setattr__(operation, "domain", input_carriers)
        object.__setattr__(operation, "totality", "total")
        object.__setattr__(operation, "bounds", None)
        return operation

    @classmethod
    def from_table(
        cls,
        symbol: OperationSymbol,
        carriers: CarrierBindings,
        table: TableEntries,
    ) -> Operation:
        """Validate one complete total table and freeze its canonical cells."""
        if type(symbol) is not OperationSymbol:
            raise OperationDefinitionError(
                field="symbol", reason="must be an exact OperationSymbol"
            )
        bindings = _snapshot_carrier_bindings(carriers, symbol)
        input_carriers = tuple(
            _carrier_for_sort(bindings, sort) for sort in symbol.inputs
        )
        output_carrier = _carrier_for_sort(bindings, symbol.output)
        source_entries = _snapshot_source(table, field="table")
        outputs: dict[tuple[int, ...], object] = {}
        declared_indices: dict[tuple[int, ...], int] = {}

        for entry_index, declaration in enumerate(source_entries):
            raw_inputs, raw_output = _require_pair(
                declaration, field="table", entry_index=entry_index
            )
            if type(raw_inputs) is not tuple:
                raise OperationDefinitionError(
                    field="table",
                    reason="input key must be an exact tuple",
                    entry_index=entry_index,
                )
            if len(raw_inputs) != symbol.arity:
                raise OperationDefinitionError(
                    field="table",
                    reason="input tuple has wrong arity",
                    entry_index=entry_index,
                )
            key = tuple(
                _canonical_table_member_index(
                    carrier,
                    argument,
                    entry_index=entry_index,
                    argument_index=argument_index,
                )
                for argument_index, (carrier, argument) in enumerate(
                    zip(input_carriers, raw_inputs, strict=True)
                )
            )
            prior_entry = declared_indices.get(key)
            if prior_entry is not None:
                raise OperationDefinitionError(
                    field="table",
                    reason="duplicate table input tuple",
                    entry_index=entry_index,
                    conflicting_entry_index=prior_entry,
                    index_tuple=key,
                )
            _canonical_table_member_index(
                output_carrier,
                raw_output,
                entry_index=entry_index,
                argument_index=None,
            )
            declared_indices[key] = entry_index
            outputs[key] = raw_output

        expected_keys = tuple(
            product(*(range(len(carrier)) for carrier in input_carriers))
        )
        for key in expected_keys:
            if key not in outputs:
                raise OperationDefinitionError(
                    field="table",
                    reason="table is missing required input tuple",
                    index_tuple=key,
                )
        canonical_table = tuple((key, outputs[key]) for key in expected_keys)
        return cls._from_validated(symbol, bindings, canonical_table)

    @classmethod
    def from_callable(
        cls,
        symbol: OperationSymbol,
        carriers: CarrierBindings,
        function: Callable[..., object],
        *,
        bounds: object | None = None,
    ) -> Operation:
        """Build a lazy total operation from a raw-output exact callable.

        ``function(*arguments)`` is not called during construction.  A raw
        output member or exact ``Defined`` member returns ``Defined``; exact
        ``Indeterminate`` and ``Failed`` results are preserved.  An exact
        ``Undefined`` violates declared totality, while a callable exception
        or malformed/non-member result becomes inert ``Failed``.  ``bounds``
        is explicit metadata retained by reference like outcome payloads; this
        slice does not enforce it or infer it from a result.
        """
        del cls
        if type(symbol) is not OperationSymbol:
            raise OperationDefinitionError(
                field="symbol", reason="must be an exact OperationSymbol"
            )
        bindings = _snapshot_carrier_bindings(carriers, symbol)
        checked_function = _require_callable(function)
        return _CallableOperation._from_callable_validated(
            symbol, bindings, checked_function, bounds
        )

    def apply(self, *arguments: object) -> Defined[object] | Indeterminate | Failed:
        """Return ``Defined(output)`` after strict arity and carrier validation."""
        if len(arguments) != self.symbol.arity:
            raise OperationApplicationError(
                self.symbol,
                reason="wrong number of arguments",
                expected_arity=self.symbol.arity,
                actual_arity=len(arguments),
            )
        indices: list[int] = []
        for argument_index, (carrier, argument) in enumerate(
            zip(self.input_carriers, arguments, strict=True)
        ):
            try:
                indices.append(carrier.index(argument))
            except CarrierMemberNotFoundError as error:
                reason = "argument is not a member of its declared carrier"
                if error.reason != "item is not a member":
                    reason = "argument carrier equality comparison failed"
                raise OperationApplicationError(
                    self.symbol,
                    reason=reason,
                    argument_index=argument_index,
                    expected_sort=carrier.sort,
                ) from error
        key = tuple(indices)
        for candidate, output in self.table:
            if candidate == key:
                return Defined(output)
        raise AssertionError("validated total-operation table lost a canonical cell")


@dataclass(frozen=True, slots=True, init=False)
class PartialOperation:
    """A finite partial operation with declared and absent canonical cells.

    The ``undefined_marker`` is interpreted strictly by object identity while
    building.  It must not be identical to an output-carrier item, so every
    actual output remains representable.  A distinct output merely equal to the
    marker is a declared output and returns ``Defined``; neither hashability nor
    equality of the marker is used.  Missing Cartesian cells remain absent and
    return ``Undefined`` without changing the output carrier.
    """

    symbol: OperationSymbol
    carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...]
    input_carriers: tuple[FiniteCarrier, ...]
    output_carrier: FiniteCarrier
    table: tuple[tuple[tuple[int, ...], object], ...]
    undefined_indices: tuple[tuple[int, ...], ...]
    domain: tuple[FiniteCarrier, ...]
    totality: str
    bounds: object | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_table` instead."""
        del args, kwargs
        raise PartialOperationDefinitionError(
            field="operation",
            reason="use PartialOperation.from_table for construction",
        )

    @classmethod
    def _from_validated(
        cls,
        symbol: OperationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        table: tuple[tuple[tuple[int, ...], object], ...],
        undefined_indices: tuple[tuple[int, ...], ...],
    ) -> PartialOperation:
        """Allocate after :meth:`from_table` proved every table invariant."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        operation = object.__new__(cls)
        object.__setattr__(operation, "symbol", symbol)
        object.__setattr__(operation, "carrier_bindings", carrier_bindings)
        object.__setattr__(operation, "input_carriers", input_carriers)
        object.__setattr__(
            operation,
            "output_carrier",
            _carrier_for_sort(carrier_bindings, symbol.output),
        )
        object.__setattr__(operation, "table", table)
        object.__setattr__(operation, "undefined_indices", undefined_indices)
        object.__setattr__(operation, "domain", input_carriers)
        object.__setattr__(operation, "totality", "partial")
        object.__setattr__(operation, "bounds", None)
        return operation

    @classmethod
    def from_table(
        cls,
        symbol: OperationSymbol,
        carriers: CarrierBindings,
        table: TableEntries,
        *,
        undefined_marker: object,
    ) -> PartialOperation:
        """Freeze a finite partial table with identity-only marker semantics."""
        if type(symbol) is not OperationSymbol:
            raise PartialOperationDefinitionError(
                field="symbol", reason="must be an exact OperationSymbol"
            )
        bindings = _snapshot_carrier_bindings(
            carriers, symbol, error_type=PartialOperationDefinitionError
        )
        input_carriers = tuple(
            _carrier_for_sort(bindings, sort) for sort in symbol.inputs
        )
        output_carrier = _carrier_for_sort(bindings, symbol.output)
        for output_index, output in enumerate(output_carrier):
            if output is undefined_marker:
                raise PartialOperationDefinitionError(
                    field="undefined_marker",
                    reason="must not be identical to a declared output carrier item",
                    index_tuple=(output_index,),
                )

        source_entries = _snapshot_source(
            table, field="table", error_type=PartialOperationDefinitionError
        )
        outputs: dict[tuple[int, ...], object] = {}
        undefined: set[tuple[int, ...]] = set()
        declared_indices: dict[tuple[int, ...], int] = {}
        for entry_index, declaration in enumerate(source_entries):
            raw_inputs, raw_output = _require_pair(
                declaration,
                field="table",
                entry_index=entry_index,
                error_type=PartialOperationDefinitionError,
            )
            if type(raw_inputs) is not tuple:
                raise PartialOperationDefinitionError(
                    field="table",
                    reason="input key must be an exact tuple",
                    entry_index=entry_index,
                )
            if len(raw_inputs) != symbol.arity:
                raise PartialOperationDefinitionError(
                    field="table",
                    reason="input tuple has wrong arity",
                    entry_index=entry_index,
                )
            key = tuple(
                _canonical_table_member_index(
                    carrier,
                    argument,
                    entry_index=entry_index,
                    argument_index=argument_index,
                    error_type=PartialOperationDefinitionError,
                )
                for argument_index, (carrier, argument) in enumerate(
                    zip(input_carriers, raw_inputs, strict=True)
                )
            )
            prior_entry = declared_indices.get(key)
            if prior_entry is not None:
                raise PartialOperationDefinitionError(
                    field="table",
                    reason="duplicate table input tuple",
                    entry_index=entry_index,
                    conflicting_entry_index=prior_entry,
                    index_tuple=key,
                )
            declared_indices[key] = entry_index
            if raw_output is undefined_marker:
                undefined.add(key)
                continue
            _canonical_table_member_index(
                output_carrier,
                raw_output,
                entry_index=entry_index,
                argument_index=None,
                error_type=PartialOperationDefinitionError,
            )
            outputs[key] = raw_output

        canonical_table = tuple((key, outputs[key]) for key in sorted(outputs))
        return cls._from_validated(
            symbol, bindings, canonical_table, tuple(sorted(undefined))
        )

    @classmethod
    def from_callable(
        cls,
        symbol: OperationSymbol,
        carriers: CarrierBindings,
        function: Callable[..., object],
        *,
        bounds: object | None = None,
    ) -> PartialOperation:
        """Build a lazy partial operation from an exact outcome callable.

        ``function(*arguments)`` is not called during construction and must
        return exactly one ``EvaluationOutcome`` branch.  A returned
        ``Defined`` value is checked against the declared output carrier;
        ``Undefined``, ``Indeterminate``, and ``Failed`` are preserved
        without totalization.  Exceptions and malformed results become inert
        ``Failed`` records.  ``bounds`` is explicit metadata retained by
        reference like outcome payloads; this slice does not enforce it.
        """
        del cls
        if type(symbol) is not OperationSymbol:
            raise PartialOperationDefinitionError(
                field="symbol", reason="must be an exact OperationSymbol"
            )
        bindings = _snapshot_carrier_bindings(
            carriers, symbol, error_type=PartialOperationDefinitionError
        )
        checked_function = _require_callable(
            function, error_type=PartialOperationDefinitionError
        )
        return _CallablePartialOperation._from_callable_validated(
            symbol, bindings, checked_function, bounds
        )

    def apply(
        self, *arguments: object
    ) -> Defined[object] | Undefined | Indeterminate | Failed:
        """Return a defined output or a stable undefined-cell outcome."""
        if len(arguments) != self.symbol.arity:
            raise PartialOperationApplicationError(
                self.symbol,
                reason="wrong number of arguments",
                expected_arity=self.symbol.arity,
                actual_arity=len(arguments),
            )
        indices: list[int] = []
        for argument_index, (carrier, argument) in enumerate(
            zip(self.input_carriers, arguments, strict=True)
        ):
            try:
                indices.append(carrier.index(argument))
            except CarrierMemberNotFoundError as error:
                reason = "argument is not a member of its declared carrier"
                if error.reason != "item is not a member":
                    reason = "argument carrier equality comparison failed"
                raise PartialOperationApplicationError(
                    self.symbol,
                    reason=reason,
                    argument_index=argument_index,
                    expected_sort=carrier.sort,
                ) from error
        key = tuple(indices)
        for candidate, output in self.table:
            if candidate == key:
                return Defined(output)
        if key in self.undefined_indices:
            return Undefined("table cell explicitly marked undefined", witness=key)
        return Undefined("table cell is not declared", witness=key)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class _CallableOperation(Operation):
    """Private identity-only record backing :meth:`Operation.from_callable`."""

    function: Callable[..., object] = field(repr=False, compare=False)
    __hash__ = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`Operation.from_callable`."""
        del args, kwargs
        raise OperationDefinitionError(
            field="operation", reason="use Operation.from_callable for construction"
        )

    @classmethod
    def _from_callable_validated(
        cls,
        symbol: OperationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        function: Callable[..., object],
        bounds: object | None,
    ) -> _CallableOperation:
        """Allocate after the public factory validated the static boundary."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        operation = object.__new__(cls)
        object.__setattr__(operation, "symbol", symbol)
        object.__setattr__(operation, "carrier_bindings", carrier_bindings)
        object.__setattr__(operation, "input_carriers", input_carriers)
        object.__setattr__(
            operation,
            "output_carrier",
            _carrier_for_sort(carrier_bindings, symbol.output),
        )
        object.__setattr__(operation, "table", ())
        object.__setattr__(operation, "function", function)
        object.__setattr__(operation, "domain", input_carriers)
        object.__setattr__(operation, "totality", "total")
        object.__setattr__(operation, "bounds", bounds)
        return operation

    def __eq__(self, other: object) -> bool:
        """Use identity because independently supplied callables are incomparable."""
        return self is other

    def apply(self, *arguments: object) -> Defined[object] | Indeterminate | Failed:
        """Validate, invoke once, and translate total-callable results safely."""
        if len(arguments) != self.symbol.arity:
            raise OperationApplicationError(
                self.symbol,
                reason="wrong number of arguments",
                expected_arity=self.symbol.arity,
                actual_arity=len(arguments),
            )
        for argument_index, (carrier, argument) in enumerate(
            zip(self.input_carriers, arguments, strict=True)
        ):
            try:
                carrier.index(argument)
            except CarrierMemberNotFoundError as error:
                reason = "argument is not a member of its declared carrier"
                if error.reason != "item is not a member":
                    reason = "argument carrier equality comparison failed"
                raise OperationApplicationError(
                    self.symbol,
                    reason=reason,
                    argument_index=argument_index,
                    expected_sort=carrier.sort,
                ) from error
        try:
            result = self.function(*arguments)
        except Exception as error:
            return Failed(
                f"callable raised {_exception_label(error)}", stage="callable"
            )
        if type(result) is Defined:
            try:
                self.output_carrier.index(result.value)
            except CarrierMemberNotFoundError:
                return Failed(
                    "callable Defined result is not an output member",
                    stage="callable",
                )
            return result
        if type(result) is Indeterminate or type(result) is Failed:
            return result
        if type(result) is Undefined:
            return Failed("total callable returned Undefined", stage="callable")
        try:
            self.output_carrier.index(result)
        except CarrierMemberNotFoundError:
            return Failed("callable returned non-member output", stage="callable")
        return Defined(result)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class _CallablePartialOperation(PartialOperation):
    """Private identity-only record backing :meth:`PartialOperation.from_callable`."""

    function: Callable[..., object] = field(repr=False, compare=False)
    __hash__ = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`PartialOperation.from_callable`."""
        del args, kwargs
        raise PartialOperationDefinitionError(
            field="operation",
            reason="use PartialOperation.from_callable for construction",
        )

    @classmethod
    def _from_callable_validated(
        cls,
        symbol: OperationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        function: Callable[..., object],
        bounds: object | None,
    ) -> _CallablePartialOperation:
        """Allocate after the public factory validated the static boundary."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        operation = object.__new__(cls)
        object.__setattr__(operation, "symbol", symbol)
        object.__setattr__(operation, "carrier_bindings", carrier_bindings)
        object.__setattr__(operation, "input_carriers", input_carriers)
        object.__setattr__(
            operation,
            "output_carrier",
            _carrier_for_sort(carrier_bindings, symbol.output),
        )
        object.__setattr__(operation, "table", ())
        object.__setattr__(operation, "undefined_indices", ())
        object.__setattr__(operation, "function", function)
        object.__setattr__(operation, "domain", input_carriers)
        object.__setattr__(operation, "totality", "partial")
        object.__setattr__(operation, "bounds", bounds)
        return operation

    def __eq__(self, other: object) -> bool:
        """Use identity because independently supplied callables are incomparable."""
        return self is other

    def apply(
        self, *arguments: object
    ) -> Defined[object] | Undefined | Indeterminate | Failed:
        """Validate, invoke once, and preserve exact callable outcome branches."""
        if len(arguments) != self.symbol.arity:
            raise PartialOperationApplicationError(
                self.symbol,
                reason="wrong number of arguments",
                expected_arity=self.symbol.arity,
                actual_arity=len(arguments),
            )
        for argument_index, (carrier, argument) in enumerate(
            zip(self.input_carriers, arguments, strict=True)
        ):
            try:
                carrier.index(argument)
            except CarrierMemberNotFoundError as error:
                reason = "argument is not a member of its declared carrier"
                if error.reason != "item is not a member":
                    reason = "argument carrier equality comparison failed"
                raise PartialOperationApplicationError(
                    self.symbol,
                    reason=reason,
                    argument_index=argument_index,
                    expected_sort=carrier.sort,
                ) from error
        try:
            result = self.function(*arguments)
        except Exception as error:
            return Failed(
                f"callable raised {_exception_label(error)}", stage="callable"
            )
        if type(result) is Defined:
            try:
                self.output_carrier.index(result.value)
            except CarrierMemberNotFoundError:
                return Failed(
                    "callable Defined result is not an output member",
                    stage="callable",
                )
            return result
        if type(result) is Undefined:
            return result
        if type(result) is Indeterminate:
            return result
        if type(result) is Failed:
            return result
        return Failed("callable returned invalid evaluation outcome", stage="callable")
