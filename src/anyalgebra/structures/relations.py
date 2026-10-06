"""Immutable finite relations backed by extensions or lazy predicates.

Relations are deliberately separate from operations: a table is a finite
extension whose declared tuples hold and whose other well-typed tuples do not.
Predicate relations preserve bounded/failure outcomes without turning them into
truth values.  Both construction routes accept ordered bindings or mappings,
snapshot their inputs once, and support unhashable carrier members by storing
canonical carrier-index tuples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import CarrierMemberNotFoundError, FiniteCarrier, Sort
from anyalgebra.structures.outcomes import Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import RelationSymbol


class RelationDefinitionError(AnyAlgebraError, ValueError):
    """A relation declaration was malformed at the static boundary."""

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
        """Retain address-free declaration coordinates and safe metadata."""
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
        detail = f"invalid relation {location}: {reason}"
        if conflicting_binding_index is not None:
            detail = f"{detail} (conflicts with binding {conflicting_binding_index})"
        if conflicting_entry_index is not None:
            detail = f"{detail} (conflicts with entry {conflicting_entry_index})"
        super().__init__(detail)


class RelationApplicationError(AnyAlgebraError, ValueError):
    """A relation query violated its declared arity or input carriers."""

    def __init__(
        self,
        symbol: RelationSymbol,
        *,
        reason: str,
        argument_index: int | None = None,
        expected_arity: int | None = None,
        actual_arity: int | None = None,
        expected_sort: Sort | None = None,
    ) -> None:
        """Keep relation diagnostics typed while never rendering arguments."""
        self.symbol = symbol
        self.reason = reason
        self.argument_index = argument_index
        self.expected_arity = expected_arity
        self.actual_arity = actual_arity
        self.expected_sort = expected_sort
        location = (
            "arguments" if argument_index is None else f"argument[{argument_index}]"
        )
        super().__init__(
            f"invalid application of relation {symbol.name!r} {location}: {reason}"
        )


@dataclass(frozen=True, slots=True, init=False)
class RelationResult:
    """The exact Boolean result of a successfully decided relation query."""

    holds: bool

    def __init__(self, holds: bool) -> None:
        """Require an exact built-in Boolean rather than generic truthiness."""
        if type(holds) is not bool:
            raise TypeError("relation result holds must be an exact built-in bool")
        object.__setattr__(self, "holds", holds)

    def __bool__(self) -> bool:
        """Expose the decided truth value only at an explicit Boolean boundary."""
        return self.holds


CarrierBindings = Iterable[tuple[Sort, FiniteCarrier]] | Mapping[Sort, FiniteCarrier]
TupleEntries = Iterable[tuple[object, ...]] | Mapping[tuple[object, ...], object]


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


def _snapshot_source(values: object, *, field: str) -> tuple[object, ...]:
    """Consume an iterable or mapping-items view exactly once."""
    if isinstance(values, Mapping):
        values = values.items()
    if isinstance(values, str | bytes):
        raise RelationDefinitionError(
            field=field, reason="must be an iterable of declared entries"
        )
    try:
        return tuple(values)  # type: ignore[arg-type]
    except (TypeError, RuntimeError) as error:
        raise RelationDefinitionError(
            field=field, reason="could not be snapshotted as an iterable"
        ) from error


def _require_pair(
    value: object,
    *,
    field: str,
    binding_index: int | None = None,
) -> tuple[object, object]:
    """Validate a concrete two-item binding without iterating it twice."""
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise RelationDefinitionError(
            field=field, reason="entry must be a pair", binding_index=binding_index
        )
    return value[0], value[1]


def _required_sorts(symbol: RelationSymbol) -> tuple[Sort, ...]:
    """Return required input sorts in first-occurrence order."""
    required: list[Sort] = []
    for sort in symbol.inputs:
        if sort not in required:
            required.append(sort)
    return tuple(required)


def _snapshot_carrier_bindings(
    values: object, symbol: RelationSymbol
) -> tuple[tuple[Sort, FiniteCarrier], ...]:
    """Validate one complete exact finite-carrier binding per needed sort."""
    snapshot = _snapshot_source(values, field="carriers")
    bindings: list[tuple[Sort, FiniteCarrier]] = []
    for index, declaration in enumerate(snapshot):
        raw_sort, raw_carrier = _require_pair(
            declaration, field="carriers", binding_index=index
        )
        if type(raw_sort) is not Sort:
            raise RelationDefinitionError(
                field="carriers",
                reason="binding must contain an exact Sort",
                binding_index=index,
            )
        if type(raw_carrier) is not FiniteCarrier:
            raise RelationDefinitionError(
                field="carriers",
                reason="binding must contain an exact FiniteCarrier",
                binding_index=index,
            )
        if raw_carrier.sort != raw_sort:
            raise RelationDefinitionError(
                field="carriers",
                reason="carrier sort does not match binding sort",
                binding_index=index,
                expected_sort=raw_sort,
            )
        for prior_index, (prior_sort, _) in enumerate(bindings):
            if prior_sort == raw_sort:
                raise RelationDefinitionError(
                    field="carriers",
                    reason="duplicate carrier binding",
                    binding_index=index,
                    conflicting_binding_index=prior_index,
                    expected_sort=raw_sort,
                )
        bindings.append((raw_sort, raw_carrier))
    for sort in _required_sorts(symbol):
        if not any(bound_sort == sort for bound_sort, _ in bindings):
            raise RelationDefinitionError(
                field="carriers",
                reason="missing required carrier binding",
                expected_sort=sort,
            )
    return tuple(bindings)


def _carrier_for_sort(
    bindings: tuple[tuple[Sort, FiniteCarrier], ...], sort: Sort
) -> FiniteCarrier:
    """Look up a carrier after complete bindings were already validated."""
    for bound_sort, carrier in bindings:
        if bound_sort == sort:
            return carrier
    raise AssertionError("validated relation is missing a required carrier")


@dataclass(frozen=True, slots=True, init=False)
class Relation:
    """A finite extension relation with false for every absent valid tuple."""

    symbol: RelationSymbol
    carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...]
    input_carriers: tuple[FiniteCarrier, ...]
    tuples: tuple[tuple[int, ...], ...]
    domain: tuple[FiniteCarrier, ...]
    bounds: object | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_tuples` instead."""
        del args, kwargs
        raise RelationDefinitionError(
            field="relation", reason="use Relation.from_tuples for construction"
        )

    @classmethod
    def _from_validated(
        cls,
        symbol: RelationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        tuples: tuple[tuple[int, ...], ...],
    ) -> Relation:
        """Allocate an extension relation after static validation succeeds."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        relation = object.__new__(cls)
        object.__setattr__(relation, "symbol", symbol)
        object.__setattr__(relation, "carrier_bindings", carrier_bindings)
        object.__setattr__(relation, "input_carriers", input_carriers)
        object.__setattr__(relation, "tuples", tuples)
        object.__setattr__(relation, "domain", input_carriers)
        object.__setattr__(relation, "bounds", None)
        return relation

    @classmethod
    def from_tuples(
        cls, symbol: RelationSymbol, carriers: CarrierBindings, tuples: TupleEntries
    ) -> Relation:
        """Build a relation true exactly on its declared finite extension."""
        if type(symbol) is not RelationSymbol:
            raise RelationDefinitionError(
                field="symbol", reason="must be an exact RelationSymbol"
            )
        bindings = _snapshot_carrier_bindings(carriers, symbol)
        input_carriers = tuple(
            _carrier_for_sort(bindings, sort) for sort in symbol.inputs
        )
        source_entries = _snapshot_source(
            tuples.keys() if isinstance(tuples, Mapping) else tuples, field="tuples"
        )
        declared: dict[tuple[int, ...], int] = {}
        for entry_index, raw_inputs in enumerate(source_entries):
            if type(raw_inputs) is not tuple:
                raise RelationDefinitionError(
                    field="tuples",
                    reason="input tuple must be an exact tuple",
                    entry_index=entry_index,
                )
            if len(raw_inputs) != symbol.arity:
                raise RelationDefinitionError(
                    field="tuples",
                    reason="input tuple has wrong arity",
                    entry_index=entry_index,
                )
            indices: list[int] = []
            for argument_index, (carrier, argument) in enumerate(
                zip(input_carriers, raw_inputs, strict=True)
            ):
                try:
                    indices.append(carrier.index(argument))
                except CarrierMemberNotFoundError as error:
                    reason = "input is not a member of its declared carrier"
                    if error.reason != "item is not a member":
                        reason = "input carrier equality comparison failed"
                    raise RelationDefinitionError(
                        field="tuples",
                        reason=reason,
                        entry_index=entry_index,
                        argument_index=argument_index,
                        expected_sort=carrier.sort,
                    ) from error
            key = tuple(indices)
            prior_entry = declared.get(key)
            if prior_entry is not None:
                raise RelationDefinitionError(
                    field="tuples",
                    reason="duplicate relation input tuple",
                    entry_index=entry_index,
                    conflicting_entry_index=prior_entry,
                    index_tuple=key,
                )
            declared[key] = entry_index
        return cls._from_validated(symbol, bindings, tuple(sorted(declared)))

    @classmethod
    def from_predicate(
        cls,
        symbol: RelationSymbol,
        carriers: CarrierBindings,
        predicate: Callable[..., object],
        *,
        bounds: object | None = None,
    ) -> Relation:
        """Build a lazy relation predicate without executing it at construction."""
        del cls
        if type(symbol) is not RelationSymbol:
            raise RelationDefinitionError(
                field="symbol", reason="must be an exact RelationSymbol"
            )
        bindings = _snapshot_carrier_bindings(carriers, symbol)
        if not callable(predicate):
            raise RelationDefinitionError(field="predicate", reason="must be callable")
        return _CallableRelation._from_callable_validated(
            symbol, bindings, predicate, bounds
        )

    def _validated_indices(self, arguments: tuple[object, ...]) -> tuple[int, ...]:
        """Validate one query and return its carrier-index tuple."""
        if len(arguments) != self.symbol.arity:
            raise RelationApplicationError(
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
                raise RelationApplicationError(
                    self.symbol,
                    reason=reason,
                    argument_index=argument_index,
                    expected_sort=carrier.sort,
                ) from error
        return tuple(indices)

    def apply(self, *arguments: object) -> RelationResult | Indeterminate | Failed:
        """Return exact truth for one valid query to the finite extension."""
        return RelationResult(self._validated_indices(arguments) in self.tuples)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class _CallableRelation(Relation):
    """Private identity-only record backing :meth:`Relation.from_predicate`."""

    predicate: Callable[..., object] = field(repr=False, compare=False)
    __hash__ = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; only the public factory validates it."""
        del args, kwargs
        raise RelationDefinitionError(
            field="relation", reason="use Relation.from_predicate for construction"
        )

    @classmethod
    def _from_callable_validated(
        cls,
        symbol: RelationSymbol,
        carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...],
        predicate: Callable[..., object],
        bounds: object | None,
    ) -> _CallableRelation:
        """Allocate after the public factory validates static metadata."""
        input_carriers = tuple(
            _carrier_for_sort(carrier_bindings, sort) for sort in symbol.inputs
        )
        relation = object.__new__(cls)
        object.__setattr__(relation, "symbol", symbol)
        object.__setattr__(relation, "carrier_bindings", carrier_bindings)
        object.__setattr__(relation, "input_carriers", input_carriers)
        object.__setattr__(relation, "tuples", ())
        object.__setattr__(relation, "domain", input_carriers)
        object.__setattr__(relation, "bounds", bounds)
        object.__setattr__(relation, "predicate", predicate)
        return relation

    def __eq__(self, other: object) -> bool:
        """Treat opaque predicate implementations as identity-only values."""
        return self is other

    def apply(self, *arguments: object) -> RelationResult | Indeterminate | Failed:
        """Validate, invoke once, and safely normalize predicate results."""
        self._validated_indices(arguments)
        try:
            result = self.predicate(*arguments)
        except Exception as error:
            return Failed(
                f"predicate raised {_exception_label(error)}", stage="predicate"
            )
        if type(result) is bool:
            return RelationResult(result)
        if type(result) is RelationResult:
            return result
        if type(result) is Indeterminate or type(result) is Failed:
            return result
        if type(result) is Undefined:
            return Failed("predicate returned Undefined", stage="predicate")
        return Failed("predicate returned invalid relation result", stage="predicate")
