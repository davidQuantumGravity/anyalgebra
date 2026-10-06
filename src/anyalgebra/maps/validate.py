"""Exact finite validation of many-sorted morphism and isomorphism records.

The functions in this module deliberately inspect every operation and relation
of a morphism's common signature.  ``Morphism.preserved_*`` is inert claim
metadata from the map-record seam; it never narrows a validation theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal, TypeAlias

from anyalgebra.core.parents import CarrierMemberNotFoundError, FiniteCarrier, Sort
from anyalgebra.maps.core import Morphism
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.relations import Relation, RelationResult
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol
from anyalgebra.structures.structure import Structure


_SCOPE_DESCRIPTION = "all finite tuples over the literal source carriers"


def _require_exact_tuple(values: object, item_type: type[object], field: str) -> None:
    """Reject malformed public report metadata at the immutable boundary."""
    if type(values) is not tuple or any(
        type(value) is not item_type for value in values
    ):
        raise TypeError(f"{field} must be a tuple of exact {item_type.__name__} values")


def _require_count(value: object, field: str) -> None:
    """Require one non-negative exact built-in count."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative built-in int")


@dataclass(frozen=True, slots=True)
class ValidationScope:
    """The finite declaration scope covered by one validation report."""

    operation_symbols: tuple[OperationSymbol, ...]
    relation_symbols: tuple[RelationSymbol, ...]
    inverse_sorts: tuple[Sort, ...] = ()
    description: str = _SCOPE_DESCRIPTION
    source_carrier_sizes: tuple[int, ...] = ()
    target_carrier_sizes: tuple[int, ...] = ()
    algorithm: str = "finite-structure-map-v1"
    enumeration_order: str = "signature then literal carrier declaration order"
    source: Structure | None = field(default=None, repr=False)
    target: Structure | None = field(default=None, repr=False)
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Keep direct public construction as strict as generated reports."""
        _require_exact_tuple(
            self.operation_symbols, OperationSymbol, "operation_symbols"
        )
        _require_exact_tuple(self.relation_symbols, RelationSymbol, "relation_symbols")
        _require_exact_tuple(self.inverse_sorts, Sort, "inverse_sorts")
        _require_exact_tuple(self.source_carrier_sizes, int, "source_carrier_sizes")
        _require_exact_tuple(self.target_carrier_sizes, int, "target_carrier_sizes")
        for size in (*self.source_carrier_sizes, *self.target_carrier_sizes):
            _require_count(size, "carrier size")
        for field_name, value in (
            ("description", self.description),
            ("algorithm", self.algorithm),
            ("enumeration_order", self.enumeration_order),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty trimmed built-in str"
                )
        if self.source is not None and type(self.source) is not Structure:
            raise TypeError("source must be an exact Structure or None")
        if self.target is not None and type(self.target) is not Structure:
            raise TypeError("target must be an exact Structure or None")


@dataclass(frozen=True, slots=True)
class ValidationCounts:
    """Expected and evaluated finite cells, separately for each check kind."""

    operation_expected: int
    operation_evaluated: int
    relation_expected: int
    relation_evaluated: int
    inverse_expected: int = 0
    inverse_evaluated: int = 0
    mapping_expected: int = 0
    mapping_evaluated: int = 0
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Reject impossible direct counts before a report can use them."""
        for field_name, value in (
            ("operation_expected", self.operation_expected),
            ("operation_evaluated", self.operation_evaluated),
            ("relation_expected", self.relation_expected),
            ("relation_evaluated", self.relation_evaluated),
            ("inverse_expected", self.inverse_expected),
            ("inverse_evaluated", self.inverse_evaluated),
            ("mapping_expected", self.mapping_expected),
            ("mapping_evaluated", self.mapping_evaluated),
        ):
            _require_count(value, field_name)
        for expected, evaluated, count_kind in (
            (self.operation_expected, self.operation_evaluated, "operation"),
            (self.relation_expected, self.relation_evaluated, "relation"),
            (self.inverse_expected, self.inverse_evaluated, "inverse"),
            (self.mapping_expected, self.mapping_evaluated, "mapping"),
        ):
            if evaluated > expected:
                raise ValueError(f"{count_kind} evaluated count exceeds expected count")


@dataclass(frozen=True, slots=True)
class Proved:
    """An exhaustive finite validation succeeded over its recorded scope."""

    scope: ValidationScope
    counts: ValidationCounts
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """A positive report must have exhausted every recorded finite count."""
        if (
            type(self.scope) is not ValidationScope
            or type(self.counts) is not ValidationCounts
        ):
            raise TypeError("proved reports require exact scope and counts")
        if (
            self.counts.operation_expected != self.counts.operation_evaluated
            or self.counts.relation_expected != self.counts.relation_evaluated
            or self.counts.inverse_expected != self.counts.inverse_evaluated
            or self.counts.mapping_expected != self.counts.mapping_evaluated
        ):
            raise ValueError("proved reports require exhaustive evaluated counts")

    @property
    def operation_cases(self) -> int:
        """Compatibility count for evaluated operation cells."""
        return self.counts.operation_evaluated

    @property
    def relation_cases(self) -> int:
        """Compatibility count for evaluated relation cells."""
        return self.counts.relation_evaluated

    @property
    def inverse_cases(self) -> int:
        """Compatibility count for evaluated inverse cells."""
        return self.counts.inverse_evaluated

    @property
    def mapping_cases(self) -> int:
        """Number of finite component-map inputs evaluated in this session."""
        return self.counts.mapping_evaluated


@dataclass(frozen=True, slots=True)
class Counterexample:
    """The earliest deterministic witness encountered by a finite check."""

    kind: str
    symbol: OperationSymbol | RelationSymbol | None
    sort: Sort | None
    inputs: tuple[object, ...] = field(repr=False)
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Require inert deterministic witness metadata without item reprs."""
        if (
            type(self.kind) is not str
            or not self.kind
            or self.kind != self.kind.strip()
        ):
            raise ValueError(
                "counterexample kind must be a non-empty trimmed built-in str"
            )
        if self.symbol is not None and type(self.symbol) not in (
            OperationSymbol,
            RelationSymbol,
        ):
            raise TypeError("counterexample symbol must be an exact declared symbol")
        if self.sort is not None and type(self.sort) is not Sort:
            raise TypeError("counterexample sort must be an exact Sort")
        if type(self.inputs) is not tuple:
            raise TypeError("counterexample inputs must be an exact tuple")


@dataclass(frozen=True, slots=True)
class Disproved:
    """A finite validation found its earliest deterministic counterexample."""

    counterexample: Counterexample
    scope: ValidationScope
    counts: ValidationCounts
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Ensure a negative result cannot contain malformed report records."""
        if (
            type(self.counterexample) is not Counterexample
            or type(self.scope) is not ValidationScope
            or type(self.counts) is not ValidationCounts
        ):
            raise TypeError(
                "disproved reports require exact witness, scope, and counts"
            )

    @property
    def operation_cases(self) -> int:
        """Compatibility count for evaluated operation cells."""
        return self.counts.operation_evaluated

    @property
    def relation_cases(self) -> int:
        """Compatibility count for evaluated relation cells."""
        return self.counts.relation_evaluated

    @property
    def inverse_cases(self) -> int:
        """Compatibility count for evaluated inverse cells."""
        return self.counts.inverse_evaluated

    @property
    def mapping_cases(self) -> int:
        """Number of finite component-map inputs evaluated in this session."""
        return self.counts.mapping_evaluated


@dataclass(frozen=True, slots=True)
class Inconclusive:
    """The finite scope was exhausted, but at least one cell was undecidable."""

    reason: str
    scope: ValidationScope
    counts: ValidationCounts
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Keep undecidable reports inert and structurally well formed."""
        if (
            type(self.reason) is not str
            or not self.reason
            or self.reason != self.reason.strip()
        ):
            raise ValueError(
                "inconclusive reason must be a non-empty trimmed built-in str"
            )
        if (
            type(self.scope) is not ValidationScope
            or type(self.counts) is not ValidationCounts
        ):
            raise TypeError("inconclusive reports require exact scope and counts")

    @property
    def operation_cases(self) -> int:
        """Compatibility count for evaluated operation cells."""
        return self.counts.operation_evaluated

    @property
    def relation_cases(self) -> int:
        """Compatibility count for evaluated relation cells."""
        return self.counts.relation_evaluated

    @property
    def inverse_cases(self) -> int:
        """Compatibility count for evaluated inverse cells."""
        return self.counts.inverse_evaluated

    @property
    def mapping_cases(self) -> int:
        """Number of finite component-map inputs evaluated in this session."""
        return self.counts.mapping_evaluated


ValidationReport: TypeAlias = Proved | Disproved | Inconclusive


@dataclass(frozen=True, slots=True)
class _Image:
    """One cached component-image result, indexed by source carrier position."""

    state: Literal["defined", "invalid", "inconclusive"]
    value: object | None = None
    reason: str | None = None


class _ImageCache:
    """Evaluate each finite component input at most once per validation session."""

    __slots__ = ("_evaluated", "_images", "_morphism")

    def __init__(self, morphism: Morphism) -> None:
        self._morphism = morphism
        self._evaluated = 0
        self._images: tuple[list[_Image | None], ...] = tuple(
            [None] * len(carrier.items) for carrier in morphism.source.carriers
        )

    def image(self, sort_index: int, item_index: int) -> _Image:
        """Return the cached image of one literal source-carrier item."""
        cached = self._images[sort_index][item_index]
        if cached is not None:
            return cached
        self._evaluated += 1
        component = self._morphism.components[sort_index]
        source_item = self._morphism.source.carriers[sort_index].items[item_index]
        try:
            result = component.function(source_item)
        except Exception as error:
            image = _Image(
                "inconclusive",
                reason=f"component callable raised {_exception_name(error)}",
            )
        else:
            image = _normalise_image(result, self._morphism.target.carriers[sort_index])
        self._images[sort_index][item_index] = image
        return image

    @property
    def evaluated(self) -> int:
        """Return the number of distinct inputs actually invoked so far."""
        return self._evaluated


def _exception_name(error: Exception) -> str:
    """Return only a safe type label, never arbitrary exception content."""
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _normalise_image(result: object, target: FiniteCarrier) -> _Image:
    """Classify a component result without treating partiality as a proof."""
    if type(result) is Defined:
        result = result.value
    elif type(result) in (Undefined, Indeterminate, Failed):
        return _Image(
            "inconclusive", reason=f"component returned {type(result).__name__}"
        )
    try:
        target.index(result)
    except CarrierMemberNotFoundError:
        return _Image(
            "invalid", reason="component result is outside literal target carrier"
        )
    return _Image("defined", value=result)


def _operation_cases(symbol: OperationSymbol, structure: Morphism) -> int:
    """Count source tuples for one declared operation, including nullary one."""
    total = 1
    for sort in symbol.inputs:
        total *= len(structure.source.carriers[_sort_index(structure, sort)].items)
    return total


def _relation_cases(symbol: RelationSymbol, structure: Morphism) -> int:
    """Count source tuples for one declared relation, including nullary one."""
    total = 1
    for sort in symbol.inputs:
        total *= len(structure.source.carriers[_sort_index(structure, sort)].items)
    return total


def _sort_index(morphism: Morphism, sort: Sort) -> int:
    """Find the signature-order component for an already-declared sort."""
    for index, declared in enumerate(morphism.source.signature.sorts):
        if declared == sort:
            return index
    raise AssertionError("a validated signature lost a declared sort")


def _operation_for(
    operations: tuple[Operation | PartialOperation, ...], symbol: OperationSymbol
) -> Operation | PartialOperation:
    """Locate one operation in declaration order without fabricating a morphism."""
    for operation in operations:
        if operation.symbol == symbol:
            return operation
    raise AssertionError("a frozen structure lost a declared operation")


def _relation_for(relations: tuple[Relation, ...], symbol: RelationSymbol) -> Relation:
    """Locate one relation in declaration order without fabricating a morphism."""
    for relation in relations:
        if relation.symbol == symbol:
            return relation
    raise AssertionError("a frozen structure lost a declared relation")


def _outcome_reason(prefix: str, outcome: object) -> str:
    """Describe only an outcome branch, avoiding arbitrary caller payloads."""
    if type(outcome) is Undefined:
        return f"{prefix} returned Undefined"
    if type(outcome) is Indeterminate:
        return f"{prefix} returned Indeterminate"
    if type(outcome) is Failed:
        return f"{prefix} returned Failed"
    return f"{prefix} returned an unsupported outcome"


def _apply_operation(
    operation: Operation | PartialOperation, inputs: tuple[object, ...], prefix: str
) -> tuple[bool, object | str]:
    """Apply an interpretation and normalize every non-defined branch safely."""
    try:
        outcome = operation.apply(*inputs)
    except Exception as error:
        return False, f"{prefix} application raised {_exception_name(error)}"
    if type(outcome) is Defined:
        return True, outcome.value
    return False, _outcome_reason(prefix, outcome)


def _apply_relation(
    relation: Relation, inputs: tuple[object, ...], prefix: str
) -> tuple[bool, bool | str]:
    """Apply a relation and normalize every non-decided branch safely."""
    try:
        outcome = relation.apply(*inputs)
    except Exception as error:
        return False, f"{prefix} application raised {_exception_name(error)}"
    if type(outcome) is RelationResult:
        return True, outcome.holds
    return False, _outcome_reason(prefix, outcome)


def _scope(morphism: Morphism, *, inverse: bool = False) -> ValidationScope:
    """Record the full common signature, never caller-declared claims only."""
    inverse_sorts = morphism.source.signature.sorts if inverse else ()
    return ValidationScope(
        () if inverse else morphism.source.signature.operations,
        () if inverse else morphism.source.signature.relations,
        inverse_sorts,
        source_carrier_sizes=tuple(
            len(carrier.items) for carrier in morphism.source.carriers
        ),
        target_carrier_sizes=tuple(
            len(carrier.items) for carrier in morphism.target.carriers
        ),
        source=morphism.source,
        target=morphism.target,
    )


def _counts(
    morphism: Morphism,
    operation_evaluated: int,
    relation_evaluated: int,
    *,
    inverse_expected: int = 0,
    inverse_evaluated: int = 0,
    mapping_evaluated: int | None = None,
    mapping_expected: int | None = None,
) -> ValidationCounts:
    """Build a complete expected/evaluated count snapshot."""
    if mapping_expected is None:
        mapping_expected = (
            inverse_expected
            if inverse_expected
            else sum(len(carrier.items) for carrier in morphism.source.carriers)
        )
    if mapping_evaluated is None:
        mapping_evaluated = mapping_expected
    operation_expected = (
        0
        if inverse_expected
        else sum(
            _operation_cases(symbol, morphism)
            for symbol in morphism.source.signature.operations
        )
    )
    relation_expected = (
        0
        if inverse_expected
        else sum(
            _relation_cases(symbol, morphism)
            for symbol in morphism.source.signature.relations
        )
    )
    return ValidationCounts(
        operation_expected,
        operation_evaluated,
        relation_expected,
        relation_evaluated,
        inverse_expected,
        inverse_evaluated,
        mapping_expected,
        mapping_evaluated,
    )


def _invalid_image_witness(
    morphism: Morphism, sort_index: int, item_index: int
) -> Counterexample:
    """Build a deterministic map-closure witness from canonical item positions."""
    carrier = morphism.source.carriers[sort_index]
    return Counterexample(
        "invalid_component_image", None, carrier.sort, (carrier.items[item_index],)
    )


def validate_homomorphism(morphism: Morphism) -> ValidationReport:
    """Public one-session route for exhaustive homomorphism validation."""
    return _validate_homomorphism(morphism, None)


def _validate_homomorphism(
    morphism: Morphism, supplied_cache: _ImageCache | None
) -> ValidationReport:
    """Exhaustively test all common-signature operations and relations.

    Relations are checked in the homomorphism direction: every true source
    tuple must map to a true target tuple.  A false source tuple is still
    evaluated and counted, but does not impose a target truth requirement.
    """
    if type(morphism) is not Morphism:
        return Inconclusive(
            "morphism must be an exact Morphism",
            ValidationScope((), ()),
            ValidationCounts(0, 0, 0, 0),
        )
    scope = _scope(morphism)
    cache = supplied_cache if supplied_cache is not None else _ImageCache(morphism)
    first_inconclusive: str | None = None
    operation_evaluated = 0
    relation_evaluated = 0

    # A homomorphism is first a total literal carrier map.  This pass covers
    # even a sort unused by the signature's operations and relations, and fills
    # the cache so every subsequent preservation cell reuses the same image.
    mapping_evaluated = 0
    for sort_index, carrier in enumerate(morphism.source.carriers):
        for item_index in range(len(carrier.items)):
            mapping_evaluated += 1
            image = cache.image(sort_index, item_index)
            if image.state == "invalid":
                return Disproved(
                    _invalid_image_witness(morphism, sort_index, item_index),
                    scope,
                    _counts(
                        morphism,
                        operation_evaluated,
                        relation_evaluated,
                        mapping_evaluated=mapping_evaluated,
                    ),
                )
            if image.state == "inconclusive" and first_inconclusive is None:
                first_inconclusive = image.reason

    for symbol in scope.operation_symbols:
        source_operation = _operation_for(morphism.source.operations, symbol)
        target_operation = _operation_for(morphism.target.operations, symbol)
        input_indices = tuple(_sort_index(morphism, sort) for sort in symbol.inputs)
        index_ranges = tuple(
            range(len(morphism.source.carriers[index].items)) for index in input_indices
        )
        for positions in product(*index_ranges):
            operation_evaluated += 1
            inputs = tuple(
                morphism.source.carriers[index].items[position]
                for index, position in zip(input_indices, positions, strict=True)
            )
            defined, source_value = _apply_operation(
                source_operation, inputs, "source operation"
            )
            if not defined:
                if first_inconclusive is None:
                    first_inconclusive = str(source_value)
                continue
            output_index = _sort_index(morphism, symbol.output)
            source_output_position = morphism.source.carriers[output_index].index(
                source_value
            )
            mapped_output = cache.image(output_index, source_output_position)
            if mapped_output.state == "inconclusive":
                continue
            mapped_inputs: list[object] = []
            mapping_unknown = False
            for component_index, item_index in zip(
                input_indices, positions, strict=True
            ):
                image = cache.image(component_index, item_index)
                if image.state == "inconclusive":
                    mapping_unknown = True
                    break
                mapped_inputs.append(image.value)
            if mapping_unknown:
                continue
            defined, target_value = _apply_operation(
                target_operation, tuple(mapped_inputs), "target operation"
            )
            if not defined:
                if first_inconclusive is None:
                    first_inconclusive = str(target_value)
                continue
            equal = morphism.target.carriers[output_index].index(
                mapped_output.value
            ) == morphism.target.carriers[output_index].index(target_value)
            if not equal:
                return Disproved(
                    Counterexample("operation", symbol, None, inputs),
                    scope,
                    _counts(morphism, operation_evaluated, relation_evaluated),
                )

    for relation_symbol in scope.relation_symbols:
        source_relation = _relation_for(morphism.source.relations, relation_symbol)
        target_relation = _relation_for(morphism.target.relations, relation_symbol)
        input_indices = tuple(
            _sort_index(morphism, sort) for sort in relation_symbol.inputs
        )
        index_ranges = tuple(
            range(len(morphism.source.carriers[index].items)) for index in input_indices
        )
        for positions in product(*index_ranges):
            relation_evaluated += 1
            inputs = tuple(
                morphism.source.carriers[index].items[position]
                for index, position in zip(input_indices, positions, strict=True)
            )
            decided, source_holds = _apply_relation(
                source_relation, inputs, "source relation"
            )
            if not decided:
                if first_inconclusive is None:
                    first_inconclusive = str(source_holds)
                continue
            if not source_holds:
                continue
            relation_mapped_inputs: list[object] = []
            mapping_unknown = False
            for component_index, item_index in zip(
                input_indices, positions, strict=True
            ):
                image = cache.image(component_index, item_index)
                if image.state == "inconclusive":
                    mapping_unknown = True
                    break
                relation_mapped_inputs.append(image.value)
            if mapping_unknown:
                continue
            decided, target_holds = _apply_relation(
                target_relation, tuple(relation_mapped_inputs), "target relation"
            )
            if not decided:
                if first_inconclusive is None:
                    first_inconclusive = str(target_holds)
                continue
            if not target_holds:
                return Disproved(
                    Counterexample("relation", relation_symbol, None, inputs),
                    scope,
                    _counts(morphism, operation_evaluated, relation_evaluated),
                )

    counts = _counts(morphism, operation_evaluated, relation_evaluated)
    if first_inconclusive is not None:
        return Inconclusive(first_inconclusive, scope, counts)
    return Proved(scope, counts)


def validate_inverse(morphism: Morphism, inverse: Morphism) -> ValidationReport:
    """Exhaustively test both literal-carrier component inverse identities."""
    return _validate_inverse(morphism, inverse, None, None)


def _validate_inverse(
    morphism: Morphism,
    inverse: Morphism,
    supplied_forward_cache: _ImageCache | None,
    supplied_backward_cache: _ImageCache | None,
) -> ValidationReport:
    """Run inverse identities, optionally reusing two existing map sessions."""
    if type(morphism) is not Morphism or type(inverse) is not Morphism:
        return Inconclusive(
            "maps must be exact Morphism values",
            ValidationScope((), ()),
            ValidationCounts(0, 0, 0, 0),
        )
    scope = _scope(morphism, inverse=True)
    expected = sum(len(carrier.items) for carrier in morphism.source.carriers)
    expected += sum(len(carrier.items) for carrier in morphism.target.carriers)
    if morphism.source is not inverse.target or morphism.target is not inverse.source:
        return Inconclusive(
            "inverse endpoints must be literal reversed structures",
            scope,
            _counts(
                morphism,
                0,
                0,
                inverse_expected=expected,
                mapping_evaluated=0,
            ),
        )
    forward_cache = (
        supplied_forward_cache
        if supplied_forward_cache is not None
        else _ImageCache(morphism)
    )
    backward_cache = (
        supplied_backward_cache
        if supplied_backward_cache is not None
        else _ImageCache(inverse)
    )
    # Populate both finite component domains once before inverse identities so
    # every later use is deterministic and the reported mapping coverage is
    # exact even if a later identity witness stops the scan.
    for sort_index, carrier in enumerate(morphism.source.carriers):
        for item_index in range(len(carrier.items)):
            forward_cache.image(sort_index, item_index)
    for sort_index, carrier in enumerate(inverse.source.carriers):
        for item_index in range(len(carrier.items)):
            backward_cache.image(sort_index, item_index)
    first_inconclusive: str | None = None
    evaluated = 0
    for sort_index, source_carrier in enumerate(morphism.source.carriers):
        for item_index, item in enumerate(source_carrier.items):
            evaluated += 1
            forward = forward_cache.image(sort_index, item_index)
            if forward.state == "invalid":
                return Disproved(
                    _invalid_image_witness(morphism, sort_index, item_index),
                    scope,
                    _counts(
                        morphism,
                        0,
                        0,
                        inverse_expected=expected,
                        inverse_evaluated=evaluated,
                    ),
                )
            if forward.state == "inconclusive":
                if first_inconclusive is None:
                    first_inconclusive = forward.reason
                continue
            target_index = morphism.target.carriers[sort_index].index(forward.value)
            backward = backward_cache.image(sort_index, target_index)
            if backward.state == "invalid":
                return Disproved(
                    _invalid_image_witness(inverse, sort_index, target_index),
                    scope,
                    _counts(
                        morphism,
                        0,
                        0,
                        inverse_expected=expected,
                        inverse_evaluated=evaluated,
                    ),
                )
            if backward.state == "inconclusive":
                if first_inconclusive is None:
                    first_inconclusive = backward.reason
                continue
            equal = source_carrier.index(backward.value) == item_index
            if not equal:
                return Disproved(
                    Counterexample("left_inverse", None, source_carrier.sort, (item,)),
                    scope,
                    _counts(
                        morphism,
                        0,
                        0,
                        inverse_expected=expected,
                        inverse_evaluated=evaluated,
                    ),
                )
    for sort_index, target_carrier in enumerate(morphism.target.carriers):
        for item_index, item in enumerate(target_carrier.items):
            evaluated += 1
            backward = backward_cache.image(sort_index, item_index)
            if backward.state == "invalid":
                return Disproved(
                    _invalid_image_witness(inverse, sort_index, item_index),
                    scope,
                    _counts(
                        morphism,
                        0,
                        0,
                        inverse_expected=expected,
                        inverse_evaluated=evaluated,
                    ),
                )
            if backward.state == "inconclusive":
                if first_inconclusive is None:
                    first_inconclusive = backward.reason
                continue
            source_index = morphism.source.carriers[sort_index].index(backward.value)
            forward = forward_cache.image(sort_index, source_index)
            if forward.state == "inconclusive":
                continue
            equal = target_carrier.index(forward.value) == item_index
            if not equal:
                return Disproved(
                    Counterexample("right_inverse", None, target_carrier.sort, (item,)),
                    scope,
                    _counts(
                        morphism,
                        0,
                        0,
                        inverse_expected=expected,
                        inverse_evaluated=evaluated,
                    ),
                )
    counts = _counts(
        morphism, 0, 0, inverse_expected=expected, inverse_evaluated=evaluated
    )
    if first_inconclusive is not None:
        return Inconclusive(first_inconclusive, scope, counts)
    return Proved(scope, counts)


def validate_isomorphism(morphism: Morphism, inverse: Morphism) -> ValidationReport:
    """Require both directional homomorphisms and both component inverses."""
    if type(morphism) is not Morphism or type(inverse) is not Morphism:
        return Inconclusive(
            "maps must be exact Morphism values",
            ValidationScope((), ()),
            ValidationCounts(0, 0, 0, 0),
        )
    inverse_scope = _scope(morphism, inverse=True)
    inverse_expected = sum(
        len(carrier.items) for carrier in morphism.source.carriers
    ) + sum(len(carrier.items) for carrier in morphism.target.carriers)
    if morphism.source is not inverse.target or morphism.target is not inverse.source:
        return Inconclusive(
            "inverse endpoints must be literal reversed structures",
            inverse_scope,
            _counts(
                morphism,
                0,
                0,
                inverse_expected=inverse_expected,
                mapping_evaluated=0,
            ),
        )
    forward_cache = _ImageCache(morphism)
    backward_cache = _ImageCache(inverse)
    forward = _validate_homomorphism(morphism, forward_cache)
    backward = _validate_homomorphism(inverse, backward_cache)
    inverse_report = _validate_inverse(morphism, inverse, forward_cache, backward_cache)
    reports = (forward, backward, inverse_report)
    for report in reports:
        if type(report) is Disproved:
            return report
    for report in reports:
        if type(report) is Inconclusive:
            return report
    assert type(forward) is Proved
    assert type(backward) is Proved
    assert type(inverse_report) is Proved
    scope = ValidationScope(
        forward.scope.operation_symbols,
        forward.scope.relation_symbols,
        inverse_report.scope.inverse_sorts,
        "all finite tuples over both literal carrier directions",
        source_carrier_sizes=forward.scope.source_carrier_sizes,
        target_carrier_sizes=forward.scope.target_carrier_sizes,
        source=morphism.source,
        target=morphism.target,
    )
    return Proved(
        scope,
        ValidationCounts(
            forward.counts.operation_expected + backward.counts.operation_expected,
            forward.counts.operation_evaluated + backward.counts.operation_evaluated,
            forward.counts.relation_expected + backward.counts.relation_expected,
            forward.counts.relation_evaluated + backward.counts.relation_evaluated,
            inverse_report.counts.inverse_expected,
            inverse_report.counts.inverse_evaluated,
            forward.counts.mapping_expected + backward.counts.mapping_expected,
            forward.counts.mapping_evaluated + backward.counts.mapping_evaluated,
        ),
    )
