"""Immutable many-sorted operation and relation signature symbols.

The declarations here are syntactic only. They do not assign carriers, tables,
callables, truth values, or evaluation semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import Sort


class SymbolDefinitionError(AnyAlgebraError, ValueError):
    """A signature symbol received malformed declaration metadata."""

    def __init__(
        self,
        *,
        kind: str,
        field: str,
        reason: str,
        index: int | None = None,
    ) -> None:
        """Record an address-free, structured signature-definition failure."""
        self.kind = kind
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid {kind} symbol {location}: {reason}")


def _definition_error(
    kind: str,
    field_name: str,
    reason: str,
    *,
    index: int | None = None,
) -> SymbolDefinitionError:
    """Construct one deterministic structured validation diagnostic."""
    return SymbolDefinitionError(
        kind=kind,
        field=field_name,
        reason=reason,
        index=index,
    )


class SignatureDefinitionError(AnyAlgebraError, ValueError):
    """An immutable signature received malformed declaration metadata.

    Diagnostics identify declaration positions but never render caller-supplied
    values, preserving deterministic, address-free failures.
    """

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        conflicting_index: int | None = None,
        sort_index: int | None = None,
        symbol_field: str | None = None,
    ) -> None:
        """Record one structured signature-definition failure."""
        self.field = field
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        self.sort_index = sort_index
        self.symbol_field = symbol_field
        location = field if index is None else f"{field}[{index}]"
        if symbol_field is not None:
            location = f"{location}.{symbol_field}"
        if sort_index is not None:
            location = f"{location}[{sort_index}]"
        detail = f"invalid signature {location}: {reason}"
        if conflicting_index is not None:
            assert index is not None
            detail = (
                f"invalid signature {field}: {reason} at declared indices "
                f"{conflicting_index} and {index}"
            )
        super().__init__(detail)


def _signature_error(
    field: str,
    reason: str,
    *,
    index: int | None = None,
    conflicting_index: int | None = None,
    sort_index: int | None = None,
    symbol_field: str | None = None,
) -> SignatureDefinitionError:
    """Construct one deterministic structured signature diagnostic."""
    return SignatureDefinitionError(
        field=field,
        reason=reason,
        index=index,
        conflicting_index=conflicting_index,
        sort_index=sort_index,
        symbol_field=symbol_field,
    )


def _require_name(kind: str, value: object) -> str:
    """Validate a canonical exact symbol name."""
    if type(value) is not str or not value or value != value.strip():
        raise _definition_error(
            kind,
            "name",
            "must be a non-empty trimmed built-in str",
        )
    return value


def _require_notation(value: object) -> str | None:
    """Validate optional operation-notation metadata."""
    if value is None:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise _definition_error(
            "operation",
            "notation",
            "must be None or a non-empty trimmed built-in str",
        )
    return value


def _snapshot_inputs(kind: str, values: Iterable[Sort]) -> tuple[Sort, ...]:
    """Take exactly one ordered immutable input-sort snapshot."""
    if isinstance(values, str | bytes):
        raise _definition_error(
            kind,
            "inputs",
            "must be an iterable of Sort values",
        )

    try:
        inputs = tuple(values)
    except (TypeError, RuntimeError) as error:
        raise _definition_error(
            kind,
            "inputs",
            "could not be snapshotted as an iterable of Sort values",
        ) from error

    for index, sort in enumerate(inputs):
        if type(sort) is not Sort:
            raise _definition_error(
                kind,
                "inputs",
                "must contain exact Sort values",
                index=index,
            )
    return inputs


def _require_output(kind: str, value: object) -> Sort:
    """Validate an exact operation output sort."""
    if type(value) is not Sort:
        raise _definition_error(kind, "output", "must be an exact Sort")
    return value


def _snapshot_signature_collection(
    values: Iterable[object],
    *,
    field: str,
    expected_type: type[Sort] | type[OperationSymbol] | type[RelationSymbol],
    expected_description: str,
) -> tuple[object, ...]:
    """Take one immutable collection snapshot with exact-type validation."""
    if isinstance(values, str | bytes):
        raise _signature_error(
            field, f"must be an iterable of exact {expected_description} values"
        )

    try:
        snapshot = tuple(values)
    except (TypeError, RuntimeError) as error:
        raise _signature_error(
            field,
            "could not be snapshotted as an iterable of exact "
            f"{expected_description} values",
        ) from error

    for index, value in enumerate(snapshot):
        if type(value) is not expected_type:
            raise _signature_error(
                field,
                f"must contain exact {expected_description} values",
                index=index,
            )
    return snapshot


def _snapshot_sorts(values: Iterable[Sort]) -> tuple[Sort, ...]:
    """Snapshot the ordered exact-sort declarations."""
    return cast(
        tuple[Sort, ...],
        _snapshot_signature_collection(
            values,
            field="sorts",
            expected_type=Sort,
            expected_description="Sort",
        ),
    )


def _snapshot_operations(
    values: Iterable[OperationSymbol],
) -> tuple[OperationSymbol, ...]:
    """Snapshot the ordered exact-operation declarations."""
    return cast(
        tuple[OperationSymbol, ...],
        _snapshot_signature_collection(
            values,
            field="operations",
            expected_type=OperationSymbol,
            expected_description="OperationSymbol",
        ),
    )


def _snapshot_relations(values: Iterable[RelationSymbol]) -> tuple[RelationSymbol, ...]:
    """Snapshot the ordered exact-relation declarations."""
    return cast(
        tuple[RelationSymbol, ...],
        _snapshot_signature_collection(
            values,
            field="relations",
            expected_type=RelationSymbol,
            expected_description="RelationSymbol",
        ),
    )


def _reject_duplicate_sorts(sorts: tuple[Sort, ...]) -> None:
    """Reject structural duplicate sorts while preserving the supplied instances."""
    indices: dict[Sort, int] = {}
    for index, sort in enumerate(sorts):
        prior_index = indices.get(sort)
        if prior_index is not None:
            raise _signature_error(
                "sorts",
                "duplicate declared sort",
                index=index,
                conflicting_index=prior_index,
            )
        indices[sort] = index


def _reject_duplicate_symbol_names(
    symbols: tuple[OperationSymbol | RelationSymbol, ...],
    *,
    field: str,
    reason: str,
) -> None:
    """Reject same-kind symbol-name collisions regardless of declaration shape."""
    indices: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        prior_index = indices.get(symbol.name)
        if prior_index is not None:
            raise _signature_error(
                field,
                reason,
                index=index,
                conflicting_index=prior_index,
            )
        indices[symbol.name] = index


def _require_declared_sort(
    declared_sorts: frozenset[Sort],
    *,
    field: str,
    index: int,
    sort: Sort,
    sort_index: int | None,
    symbol_field: str,
) -> None:
    """Reject a symbol reference not structurally declared by the signature."""
    if sort not in declared_sorts:
        raise _signature_error(
            field,
            "references an undeclared sort",
            index=index,
            sort_index=sort_index,
            symbol_field=symbol_field,
        )


def _validate_symbol_references(
    sorts: tuple[Sort, ...],
    operations: tuple[OperationSymbol, ...],
    relations: tuple[RelationSymbol, ...],
) -> None:
    """Ensure every operation and relation reference names a declared sort."""
    declared_sorts = frozenset(sorts)
    for index, operation in enumerate(operations):
        for sort_index, sort in enumerate(operation.inputs):
            _require_declared_sort(
                declared_sorts,
                field="operations",
                index=index,
                sort=sort,
                sort_index=sort_index,
                symbol_field="inputs",
            )
        _require_declared_sort(
            declared_sorts,
            field="operations",
            index=index,
            sort=operation.output,
            sort_index=None,
            symbol_field="output",
        )
    for index, relation in enumerate(relations):
        for sort_index, sort in enumerate(relation.inputs):
            _require_declared_sort(
                declared_sorts,
                field="relations",
                index=index,
                sort=sort,
                sort_index=sort_index,
                symbol_field="inputs",
            )


@dataclass(frozen=True, slots=True, init=False)
class OperationSymbol:
    """An immutable operation declaration of arbitrary finite arity."""

    name: str
    inputs: tuple[Sort, ...]
    output: Sort
    notation: str | None = field(default=None, kw_only=True)

    def __init__(
        self,
        name: str,
        inputs: Iterable[Sort],
        output: Sort,
        *,
        notation: str | None = None,
    ) -> None:
        """Validate and freeze an operation's declaration metadata."""
        object.__setattr__(self, "name", _require_name("operation", name))
        object.__setattr__(self, "inputs", _snapshot_inputs("operation", inputs))
        object.__setattr__(self, "output", _require_output("operation", output))
        object.__setattr__(self, "notation", _require_notation(notation))

    @property
    def arity(self) -> int:
        """Return the number of ordered argument sorts."""
        return len(self.inputs)


@dataclass(frozen=True, slots=True, init=False)
class RelationSymbol:
    """An immutable relation declaration of arbitrary finite arity."""

    name: str
    inputs: tuple[Sort, ...]

    def __init__(self, name: str, inputs: Iterable[Sort]) -> None:
        """Validate and freeze a relation's declaration metadata."""
        object.__setattr__(self, "name", _require_name("relation", name))
        object.__setattr__(self, "inputs", _snapshot_inputs("relation", inputs))

    @property
    def arity(self) -> int:
        """Return the number of ordered argument sorts."""
        return len(self.inputs)


@dataclass(frozen=True, slots=True, init=False)
class Signature:
    """An immutable many-sorted syntax declaration.

    Sorts, operations, and relations retain caller declaration order and literal
    instances.  Equal sorts are duplicate declarations, while operation and
    relation names occupy separate namespaces and may share the same spelling.
    No carrier, interpretation, table, callable, or evaluation behavior belongs
    to a signature.
    """

    sorts: tuple[Sort, ...]
    operations: tuple[OperationSymbol, ...]
    relations: tuple[RelationSymbol, ...]

    def __init__(
        self,
        sorts: Iterable[Sort],
        operations: Iterable[OperationSymbol] = (),
        relations: Iterable[RelationSymbol] = (),
    ) -> None:
        """Validate all finite declaration collections before freezing state."""
        sort_snapshot = _snapshot_sorts(sorts)
        operation_snapshot = _snapshot_operations(operations)
        relation_snapshot = _snapshot_relations(relations)

        _reject_duplicate_sorts(sort_snapshot)
        _reject_duplicate_symbol_names(
            operation_snapshot,
            field="operations",
            reason="duplicate operation name",
        )
        _reject_duplicate_symbol_names(
            relation_snapshot,
            field="relations",
            reason="duplicate relation name",
        )
        _validate_symbol_references(
            sort_snapshot, operation_snapshot, relation_snapshot
        )

        object.__setattr__(self, "sorts", sort_snapshot)
        object.__setattr__(self, "operations", operation_snapshot)
        object.__setattr__(self, "relations", relation_snapshot)
