"""Generic immutable many-sorted structures and their mutable builder.

This module only assembles already-validated carriers and interpretations.  It
does not evaluate terms, apply operations, query relations, serialize records,
or assign fingerprints; those are later boundaries.  A builder accepts
registrations in any order, and each successful freeze snapshots them in the
literal declaration order owned by its ``Signature``.
"""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.relations import Relation
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature


class StructureDefinitionError(AnyAlgebraError, ValueError):
    """A structure registration or freeze boundary was malformed."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        sort: Sort | None = None,
        symbol: OperationSymbol | RelationSymbol | None = None,
    ) -> None:
        """Keep only typed declaration metadata in a stable diagnostic."""
        self.field = field
        self.reason = reason
        self.sort = sort
        self.symbol = symbol
        location = field
        if sort is not None:
            location = f"{location} sort {sort.name!r}"
        if symbol is not None:
            location = f"{location} symbol {symbol.name!r}"
        super().__init__(f"invalid structure {location}: {reason}")


def _canonical_sort(signature: Signature, candidate: Sort) -> Sort | None:
    """Return the literal Signature sort structurally equal to ``candidate``."""
    for declared in signature.sorts:
        if declared == candidate:
            return declared
    return None


def _canonical_operation(
    signature: Signature, candidate: OperationSymbol
) -> OperationSymbol | None:
    """Return the literal Signature operation symbol equal to ``candidate``."""
    for declared in signature.operations:
        if declared == candidate:
            return declared
    return None


def _canonical_relation(
    signature: Signature, candidate: RelationSymbol
) -> RelationSymbol | None:
    """Return the literal Signature relation symbol equal to ``candidate``."""
    for declared in signature.relations:
        if declared == candidate:
            return declared
    return None


def _carrier_matches(left: FiniteCarrier, right: FiniteCarrier) -> bool | None:
    """Compare carrier definitions without leaking arbitrary equality failures."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return None


@dataclass(frozen=True, slots=True, init=False)
class Structure:
    """A frozen generic model aligned to one signature's declaration order."""

    signature: Signature
    carriers: tuple[FiniteCarrier, ...]
    operations: tuple[Operation | PartialOperation, ...]
    relations: tuple[Relation, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; a builder proves all invariants."""
        del args, kwargs
        raise StructureDefinitionError(
            field="structure", reason="use StructureBuilder for construction"
        )

    @classmethod
    def _from_validated(
        cls,
        signature: Signature,
        carriers: tuple[FiniteCarrier, ...],
        operations: tuple[Operation | PartialOperation, ...],
        relations: tuple[Relation, ...],
    ) -> Structure:
        """Allocate after one builder's full immutable snapshot is validated."""
        structure = object.__new__(cls)
        object.__setattr__(structure, "signature", signature)
        object.__setattr__(structure, "carriers", carriers)
        object.__setattr__(structure, "operations", operations)
        object.__setattr__(structure, "relations", relations)
        return structure


class StructureBuilder:
    """Mutable, reusable registrations for one immutable Signature definition."""

    __slots__ = (
        "_carriers",
        "_operations",
        "_relations",
        "_signature",
    )

    def __init__(self, signature: Signature) -> None:
        """Start an empty builder bound to one exact immutable signature."""
        if type(signature) is not Signature:
            raise StructureDefinitionError(
                field="signature", reason="must be an exact Signature"
            )
        self._signature = signature
        self._carriers: dict[Sort, FiniteCarrier] = {}
        self._operations: dict[OperationSymbol, Operation | PartialOperation] = {}
        self._relations: dict[RelationSymbol, Relation] = {}

    @property
    def signature(self) -> Signature:
        """Return the immutable signature this builder is permanently bound to."""
        return self._signature

    def with_carrier(self, sort: object, carrier: object) -> StructureBuilder:
        """Register one carrier under a declared sort, without replacement."""
        if type(sort) is not Sort:
            raise StructureDefinitionError(
                field="carriers", reason="sort must be an exact Sort"
            )
        canonical = _canonical_sort(self.signature, sort)
        if canonical is None:
            raise StructureDefinitionError(
                field="carriers", reason="unknown declared sort", sort=sort
            )
        if type(carrier) is not FiniteCarrier:
            raise StructureDefinitionError(
                field="carriers",
                reason="carrier must be an exact FiniteCarrier",
                sort=canonical,
            )
        if carrier.sort != canonical:
            raise StructureDefinitionError(
                field="carriers",
                reason="carrier sort does not match registration sort",
                sort=canonical,
            )
        if canonical in self._carriers:
            raise StructureDefinitionError(
                field="carriers",
                reason="duplicate carrier registration",
                sort=canonical,
            )
        self._carriers[canonical] = carrier
        return self

    def with_operation(self, symbol: object, operation: object) -> StructureBuilder:
        """Register one declared total or partial operation interpretation."""
        if type(symbol) is not OperationSymbol:
            raise StructureDefinitionError(
                field="operations", reason="symbol must be an exact OperationSymbol"
            )
        canonical = _canonical_operation(self.signature, symbol)
        if canonical is None:
            raise StructureDefinitionError(
                field="operations",
                reason="unknown declared operation symbol",
                symbol=symbol,
            )
        if not isinstance(operation, Operation | PartialOperation):
            raise StructureDefinitionError(
                field="operations",
                reason="interpretation must be an Operation or PartialOperation",
                symbol=canonical,
            )
        if operation.symbol != canonical:
            raise StructureDefinitionError(
                field="operations",
                reason="interpretation symbol does not match registration",
                symbol=canonical,
            )
        if canonical in self._operations:
            raise StructureDefinitionError(
                field="operations",
                reason="duplicate operation registration",
                symbol=canonical,
            )
        self._validate_known_interpretation_carriers(operation, canonical, "operations")
        self._operations[canonical] = operation
        return self

    def with_relation(self, symbol: object, relation: object) -> StructureBuilder:
        """Register one declared relation interpretation."""
        if type(symbol) is not RelationSymbol:
            raise StructureDefinitionError(
                field="relations", reason="symbol must be an exact RelationSymbol"
            )
        canonical = _canonical_relation(self.signature, symbol)
        if canonical is None:
            raise StructureDefinitionError(
                field="relations",
                reason="unknown declared relation symbol",
                symbol=symbol,
            )
        if not isinstance(relation, Relation):
            raise StructureDefinitionError(
                field="relations",
                reason="interpretation must be an exact Relation",
                symbol=canonical,
            )
        if relation.symbol != canonical:
            raise StructureDefinitionError(
                field="relations",
                reason="interpretation symbol does not match registration",
                symbol=canonical,
            )
        if canonical in self._relations:
            raise StructureDefinitionError(
                field="relations",
                reason="duplicate relation registration",
                symbol=canonical,
            )
        self._validate_known_interpretation_carriers(relation, canonical, "relations")
        self._relations[canonical] = relation
        return self

    def _validate_known_interpretation_carriers(
        self,
        interpretation: Operation | PartialOperation | Relation,
        symbol: OperationSymbol | RelationSymbol,
        field: str,
    ) -> None:
        """Check retained bindings that can already be compared to a carrier."""
        for bound_sort, bound_carrier in interpretation.carrier_bindings:
            canonical_sort = _canonical_sort(self.signature, bound_sort)
            if canonical_sort is None:
                raise StructureDefinitionError(
                    field=field,
                    reason="interpretation carrier sort is not declared",
                    symbol=symbol,
                )
            declared_carrier = self._carriers.get(canonical_sort)
            if declared_carrier is None:
                continue
            matches = _carrier_matches(declared_carrier, bound_carrier)
            if matches is None:
                raise StructureDefinitionError(
                    field=field,
                    reason="interpretation carrier equality comparison failed",
                    symbol=symbol,
                )
            if not matches:
                raise StructureDefinitionError(
                    field=field,
                    reason="interpretation carrier does not match structure carrier",
                    symbol=symbol,
                )

    def freeze(self) -> Structure:
        """Validate completeness and return a new frozen immutable snapshot."""
        for sort in self.signature.sorts:
            if sort not in self._carriers:
                raise StructureDefinitionError(
                    field="carriers",
                    reason="missing required carrier registration",
                    sort=sort,
                )
        for operation_symbol in self.signature.operations:
            if operation_symbol not in self._operations:
                raise StructureDefinitionError(
                    field="operations",
                    reason="missing required operation registration",
                    symbol=operation_symbol,
                )
        for relation_symbol in self.signature.relations:
            if relation_symbol not in self._relations:
                raise StructureDefinitionError(
                    field="relations",
                    reason="missing required relation registration",
                    symbol=relation_symbol,
                )
        for operation_symbol in self.signature.operations:
            self._validate_known_interpretation_carriers(
                self._operations[operation_symbol], operation_symbol, "operations"
            )
        for relation_symbol in self.signature.relations:
            self._validate_known_interpretation_carriers(
                self._relations[relation_symbol], relation_symbol, "relations"
            )
        return Structure._from_validated(
            self.signature,
            tuple(self._carriers[sort] for sort in self.signature.sorts),
            tuple(self._operations[symbol] for symbol in self.signature.operations),
            tuple(self._relations[symbol] for symbol in self.signature.relations),
        )
