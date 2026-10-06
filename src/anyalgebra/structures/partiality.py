"""Explicit strict-bottom totalization for finite table partial operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import CarrierMemberNotFoundError, FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined


class TotalizationDefinitionError(AnyAlgebraError, ValueError):
    """A strict-bottom totalization boundary received malformed input."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Keep diagnostics typed and free of arbitrary caller reprs."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid totalization {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False)
class TotalizationEmbedding:
    """The checked identity-on-values map into one adjoined output carrier."""

    source_carrier: FiniteCarrier
    target_carrier: FiniteCarrier
    bottom: object
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; totalize builds the checked map."""
        del args, kwargs
        raise TotalizationDefinitionError(
            field="embedding", reason="use totalize for construction"
        )

    @classmethod
    def _from_validated(
        cls,
        source_carrier: FiniteCarrier,
        target_carrier: FiniteCarrier,
        bottom: object,
    ) -> TotalizationEmbedding:
        """Allocate after totalization proves carrier membership invariants."""
        embedding = object.__new__(cls)
        object.__setattr__(embedding, "source_carrier", source_carrier)
        object.__setattr__(embedding, "target_carrier", target_carrier)
        object.__setattr__(embedding, "bottom", bottom)
        return embedding

    def embed(self, value: object) -> object:
        """Embed one original output member without coercion or value rewriting."""
        try:
            self.source_carrier.index(value)
        except CarrierMemberNotFoundError as error:
            reason = "embedding value is not a source output member"
            if error.reason != "item is not a member":
                reason = "embedding value carrier equality comparison failed"
            raise TotalizationDefinitionError(
                field="embedding", reason=reason
            ) from error
        return value


@dataclass(frozen=True, slots=True, init=False)
class Totalization:
    """One strict-bottom total operation and its explicit source embedding."""

    source: PartialOperation
    operation: Operation
    source_carrier: FiniteCarrier
    totalized_carrier: FiniteCarrier
    bottom: object
    strict_policy: str
    embedding: TotalizationEmbedding
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :func:`totalize`."""
        del args, kwargs
        raise TotalizationDefinitionError(
            field="totalization", reason="use totalize for construction"
        )

    @classmethod
    def _from_validated(
        cls,
        source: PartialOperation,
        operation: Operation,
        source_carrier: FiniteCarrier,
        totalized_carrier: FiniteCarrier,
        bottom: object,
        embedding: TotalizationEmbedding,
    ) -> Totalization:
        """Allocate after the exhaustive strict-bottom table is complete."""
        result = object.__new__(cls)
        object.__setattr__(result, "source", source)
        object.__setattr__(result, "operation", operation)
        object.__setattr__(result, "source_carrier", source_carrier)
        object.__setattr__(result, "totalized_carrier", totalized_carrier)
        object.__setattr__(result, "bottom", bottom)
        object.__setattr__(result, "strict_policy", "strict-bottom")
        object.__setattr__(result, "embedding", embedding)
        return result


def _carrier_for_sort(
    bindings: tuple[tuple[Sort, FiniteCarrier], ...], sort: Sort
) -> FiniteCarrier:
    """Find one already-validated bound carrier without using hashes on values."""
    for bound_sort, carrier in bindings:
        if bound_sort == sort:
            return carrier
    raise AssertionError("partial operation is missing a validated input carrier")


def _is_callable_partial(operation: PartialOperation) -> bool:
    """Recognize the private lazy route without invoking or rendering it."""
    return hasattr(operation, "function")


def totalize(partial_operation: PartialOperation, *, bottom: object) -> Totalization:
    """Adjoin ``bottom`` to a finite table partial operation under strictness.

    Defined source cells retain their literal outputs.  Both explicit and
    omitted undefined cells become ``bottom``; any newly introduced bottom
    input in an occurrence of the output sort is also forced to ``bottom``.
    Callable partial operations are rejected because their non-defined branches
    may represent bounded or failed computation rather than undefinedness.
    """
    if not isinstance(partial_operation, PartialOperation):
        raise TotalizationDefinitionError(
            field="partial_operation",
            reason="partial operation must be an exact PartialOperation",
        )
    if _is_callable_partial(partial_operation):
        raise TotalizationDefinitionError(
            field="partial_operation",
            reason="callable partial operations cannot be totalized",
        )
    source_carrier = partial_operation.output_carrier
    try:
        source_carrier.index(bottom)
    except CarrierMemberNotFoundError as error:
        if error.reason != "item is not a member":
            raise TotalizationDefinitionError(
                field="bottom", reason="bottom carrier equality comparison failed"
            ) from error
    else:
        raise TotalizationDefinitionError(
            field="bottom", reason="bottom is already an output carrier member"
        )

    totalized_carrier = FiniteCarrier(
        (*source_carrier.items, bottom), sort=source_carrier.sort
    )
    output_sort = partial_operation.symbol.output
    bindings = tuple(
        (sort, totalized_carrier if sort == output_sort else carrier)
        for sort, carrier in partial_operation.carrier_bindings
    )
    input_carriers = tuple(
        _carrier_for_sort(bindings, sort) for sort in partial_operation.symbol.inputs
    )
    entries: list[tuple[tuple[object, ...], object]] = []
    for inputs in product(*(carrier.items for carrier in input_carriers)):
        has_bottom = any(
            sort == output_sort and value is bottom
            for sort, value in zip(partial_operation.symbol.inputs, inputs, strict=True)
        )
        if has_bottom:
            entries.append((inputs, bottom))
            continue
        outcome = partial_operation.apply(*inputs)
        entries.append((inputs, outcome.value if type(outcome) is Defined else bottom))
    operation = Operation.from_table(partial_operation.symbol, bindings, entries)
    embedding = TotalizationEmbedding._from_validated(
        source_carrier, totalized_carrier, bottom
    )
    return Totalization._from_validated(
        partial_operation,
        operation,
        source_carrier,
        totalized_carrier,
        bottom,
        embedding,
    )
