"""Standalone recursive evaluation of typed terms in frozen structures.

The public partiality vocabulary stays closed: every evaluation result contains
one exact existing ``EvaluationOutcome`` branch.  The traversal location of a
non-defined result is separate immutable metadata, expressed as a tuple of
ordered child indices from the evaluated root.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import CarrierMemberNotFoundError, FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.terms import Term, Variable


Outcome = Defined[object] | Undefined | Indeterminate | Failed


class EvaluationDefinitionError(AnyAlgebraError, ValueError):
    """An evaluator boundary received malformed structure, term, or environment."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        expected_sort: Sort | None = None,
    ) -> None:
        """Record only stable declared coordinates, never caller value reprs."""
        self.field = field
        self.reason = reason
        self.index = index
        self.expected_sort = expected_sort
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid evaluation {location}: {reason}")


@dataclass(frozen=True, slots=True, init=False)
class EvaluationResult:
    """One closed evaluation outcome plus an immutable traversal path."""

    outcome: Outcome
    path: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, outcome: Outcome, path: tuple[int, ...] = ()) -> None:
        """Require an exact outcome branch and address-free index path."""
        if type(outcome) not in (Defined, Undefined, Indeterminate, Failed):
            raise TypeError(
                "evaluation result outcome must be an exact EvaluationOutcome"
            )
        if type(path) is not tuple:
            raise TypeError("evaluation result path must be an exact tuple")
        for index in path:
            if type(index) is not int or index < 0:
                raise TypeError(
                    "evaluation result path entries must be non-negative built-in int"
                )
        if type(outcome) is Defined and path:
            raise TypeError("a Defined evaluation result must have an empty path")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "path", path)


def _carrier_for_sort(structure: Structure, sort: Sort) -> FiniteCarrier | None:
    """Find a signature-owned carrier by literal declaration position."""
    for declared, carrier in zip(
        structure.signature.sorts, structure.carriers, strict=True
    ):
        if declared == sort:
            return carrier
    return None


def _snapshot_environment(
    structure: Structure, environment: object
) -> tuple[tuple[Variable, object], ...]:
    """Validate a mapping snapshot while permitting valid declared-sort extras."""
    if not isinstance(environment, Mapping):
        raise EvaluationDefinitionError(
            field="environment", reason="environment must be a mapping"
        )
    try:
        raw_entries = tuple(environment.items())
    except (TypeError, RuntimeError) as error:
        raise EvaluationDefinitionError(
            field="environment", reason="environment could not be snapshotted"
        ) from error

    entries: list[tuple[Variable, object]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise EvaluationDefinitionError(
                field="environment",
                reason="environment entries must be pairs",
                index=index,
            )
        variable, value = entry
        if type(variable) is not Variable:
            raise EvaluationDefinitionError(
                field="environment",
                reason="environment keys must be exact Variable values",
                index=index,
            )
        carrier = _carrier_for_sort(structure, variable.sort)
        if carrier is None:
            raise EvaluationDefinitionError(
                field="environment",
                reason="environment variable sort is not declared",
                index=index,
                expected_sort=variable.sort,
            )
        for _prior_index, (prior, _) in enumerate(entries):
            if prior == variable:
                raise EvaluationDefinitionError(
                    field="environment",
                    reason="duplicate structurally equal environment variable",
                    index=index,
                )
        try:
            carrier.index(value)
        except CarrierMemberNotFoundError as error:
            reason = "environment value is not a member of its declared carrier"
            if error.reason != "item is not a member":
                reason = "environment value carrier equality comparison failed"
            raise EvaluationDefinitionError(
                field="environment",
                reason=reason,
                index=index,
                expected_sort=variable.sort,
            ) from error
        entries.append((variable, value))
    return tuple(entries)


def _binding_for(
    entries: tuple[tuple[Variable, object], ...], variable: Variable
) -> object:
    """Resolve a free variable by safe structural equality rather than identity."""
    for candidate, value in entries:
        if candidate == variable:
            return value
    raise EvaluationDefinitionError(
        field="environment",
        reason="missing required variable binding",
        expected_sort=variable.sort,
    )


def _operation_for(
    structure: Structure, symbol: OperationSymbol
) -> Operation | PartialOperation | None:
    """Resolve only the literal signature symbol used by the term tree."""
    for declared, operation in zip(
        structure.signature.operations, structure.operations, strict=True
    ):
        if declared == symbol:
            return operation
    return None


def _preflight(
    structure: Structure,
    term: Term,
    environment: tuple[tuple[Variable, object], ...],
) -> None:
    """Resolve every free variable and symbol before any callable can run."""
    if term.variable_value is not None:
        _binding_for(environment, term.variable_value)
        return
    assert term.symbol is not None
    if _operation_for(structure, term.symbol) is None:
        raise EvaluationDefinitionError(
            field="term", reason="term operation symbol is not declared by structure"
        )
    for child in term.arguments:
        _preflight(structure, child, environment)


def _evaluate(
    structure: Structure,
    term: Term,
    environment: tuple[tuple[Variable, object], ...],
) -> EvaluationResult:
    """Evaluate one validated tree depth-first, left-to-right, exactly once."""
    if term.variable_value is not None:
        return EvaluationResult(Defined(_binding_for(environment, term.variable_value)))
    assert term.symbol is not None
    operation = _operation_for(structure, term.symbol)
    assert operation is not None
    arguments: list[object] = []
    for index, child in enumerate(term.arguments):
        child_result = _evaluate(structure, child, environment)
        if type(child_result.outcome) is not Defined:
            return EvaluationResult(child_result.outcome, (index, *child_result.path))
        arguments.append(child_result.outcome.value)
    outcome = operation.apply(*arguments)
    if type(outcome) not in (Defined, Undefined, Indeterminate, Failed):
        return EvaluationResult(
            Failed("operation returned invalid evaluation outcome", stage="evaluation")
        )
    return EvaluationResult(outcome)


def evaluate_term(
    structure: Structure, term: Term, environment: object
) -> EvaluationResult:
    """Evaluate a term without adding evaluation methods to ``Structure``.

    All supplied environment bindings are validated eagerly.  Extra bindings
    are permitted when their variables use a declared structure sort and their
    values belong to the matching carrier.  A free variable is then resolved by
    structural equality.  Operation symbols also resolve structurally to the
    interpretation in this structure's signature; a genuinely absent symbol
    cannot select a same-shaped foreign interpretation.
    """
    if type(structure) is not Structure:
        raise EvaluationDefinitionError(
            field="structure", reason="structure must be an exact Structure"
        )
    if type(term) is not Term:
        raise EvaluationDefinitionError(
            field="term", reason="term must be an exact Term"
        )
    snapshot = _snapshot_environment(structure, environment)
    _preflight(structure, term, snapshot)
    return _evaluate(structure, term, snapshot)
