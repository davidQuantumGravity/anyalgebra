"""Exact bounded exhaustive validation of immutable quantified equation laws."""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import CarrierMemberNotFoundError, FiniteCarrier, Sort
from anyalgebra.structures.evaluate import EvaluationResult, evaluate_term
from anyalgebra.structures.laws import Law
from anyalgebra.structures.outcomes import Defined, Failed, Undefined
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.terms import Term
from anyalgebra.validation.domains import (
    FiniteSubstitutionDomain,
    SubstitutionDomainError,
)


_MAX_TERM_NODES = 65_536


class ValidationDefinitionError(AnyAlgebraError, ValueError):
    """An exhaustive-law validation request failed before operation execution."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Keep only stable declaration coordinates in the diagnostic."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid exhaustive validation {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ValidationWitness:
    """One safe first-enumeration witness, retaining outcomes without rendering data."""

    substitution_indices: tuple[int, ...]
    phase: str
    premise_index: int | None
    left: EvaluationResult | None
    right: EvaluationResult | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked witness construction."""
        del args, kwargs
        raise ValidationDefinitionError(field="witness", reason="is validator-owned")

    @classmethod
    def _create(
        cls,
        indices: tuple[int, ...],
        phase: str,
        premise_index: int | None,
        left: EvaluationResult | None,
        right: EvaluationResult | None,
    ) -> ValidationWitness:
        if cls is not ValidationWitness:
            raise ValidationDefinitionError(
                field="witness", reason="factory requires exact ValidationWitness"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "substitution_indices", indices)
        object.__setattr__(value, "phase", phase)
        object.__setattr__(value, "premise_index", premise_index)
        object.__setattr__(value, "left", left)
        object.__setattr__(value, "right", right)
        return value

    def __repr__(self) -> str:
        """Render locations and outcome tags only, never arbitrary members."""
        left = None if self.left is None else type(self.left.outcome).__name__
        right = None if self.right is None else type(self.right.outcome).__name__
        return (
            "ValidationWitness("
            f"substitution_indices={self.substitution_indices}, phase={self.phase!r}, "
            f"premise_index={self.premise_index}, left_outcome={left!r}, "
            f"right_outcome={right!r})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ValidationReport:
    """Replayable bounded enumeration metadata common to all exact outcomes."""

    structure: Structure
    law: Law
    expected_assignments: int
    evaluated_assignments: int
    premise_evaluations: int
    conclusion_evaluations: int
    vacuous_assignments: int
    skipped_assignments: int
    undecidable_assignments: int
    enumeration_policy: str
    partial_semantics: str
    variable_carrier_sizes: tuple[int, ...]
    algorithm: str
    algorithm_version: int
    witness: ValidationWitness | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked reports; only ``validate_law`` determines their tag."""
        del args, kwargs
        raise ValidationDefinitionError(field="report", reason="is validator-owned")

    def __repr__(self) -> str:
        """Expose provenance-free counts and the class outcome tag safely."""
        return (
            f"{type(self).__name__}(expected_assignments={self.expected_assignments}, "
            f"evaluated_assignments={self.evaluated_assignments}, "
            f"premise_evaluations={self.premise_evaluations}, "
            f"conclusion_evaluations={self.conclusion_evaluations}, "
            f"vacuous_assignments={self.vacuous_assignments}, "
            f"skipped_assignments={self.skipped_assignments}, "
            f"undecidable_assignments={self.undecidable_assignments}, "
            f"partial_semantics={self.partial_semantics!r}, "
            f"algorithm={self.algorithm!r}, "
            f"algorithm_version={self.algorithm_version}, "
            f"has_witness={self.witness is not None})"
        )

    def __str__(self) -> str:
        """Say the outcome in words, with locations only and no member payloads."""
        name = self.law.name
        if self.witness is not None and type(self).__name__ == "Disproved":
            return (
                f"Disproved: {name} fails at assignment indices "
                f"{self.witness.substitution_indices}, found after "
                f"{self.evaluated_assignments} of {self.expected_assignments} "
                "assignments"
            )
        if self.undecidable_assignments:
            return (
                f"Inconclusive: {name} has {self.undecidable_assignments} undecidable "
                f"assignments among {self.expected_assignments}"
            )
        return f"Proved: {name} holds on all {self.expected_assignments} assignments"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Proved(ValidationReport):
    """Every bounded assignment was exhausted without a counterexample or gap."""


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Disproved(ValidationReport):
    """The first deterministic definite counterexample was found."""


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Inconclusive(ValidationReport):
    """No counterexample was found, but at least one assignment was undecidable."""


def _report(
    kind: type[Proved] | type[Disproved] | type[Inconclusive],
    structure: Structure,
    law: Law,
    *,
    expected: int,
    evaluated: int,
    premises: int,
    conclusions: int,
    vacuous: int,
    skipped: int,
    undecidable: int,
    variable_carrier_sizes: tuple[int, ...],
    witness: ValidationWitness | None,
) -> Proved | Disproved | Inconclusive:
    """Allocate a sealed exact result after all retained fields are known."""
    value = object.__new__(kind)
    object.__setattr__(value, "structure", structure)
    object.__setattr__(value, "law", law)
    object.__setattr__(value, "expected_assignments", expected)
    object.__setattr__(value, "evaluated_assignments", evaluated)
    object.__setattr__(value, "premise_evaluations", premises)
    object.__setattr__(value, "conclusion_evaluations", conclusions)
    object.__setattr__(value, "vacuous_assignments", vacuous)
    object.__setattr__(value, "skipped_assignments", skipped)
    object.__setattr__(value, "undecidable_assignments", undecidable)
    object.__setattr__(
        value, "enumeration_policy", "finite_substitution_domain_lexicographic"
    )
    object.__setattr__(value, "partial_semantics", law.partial_semantics)
    object.__setattr__(value, "variable_carrier_sizes", variable_carrier_sizes)
    object.__setattr__(value, "algorithm", "anyalgebra.exhaustive_finite_law")
    object.__setattr__(value, "algorithm_version", 1)
    object.__setattr__(value, "witness", witness)
    return value


def _carrier_for_sort(structure: Structure, sort: Sort) -> FiniteCarrier | None:
    """Return the literal declared carrier for a structurally matching sort."""
    for declared, carrier in zip(
        structure.signature.sorts, structure.carriers, strict=True
    ):
        if declared == sort:
            return carrier
    return None


def _symbols(term: Term) -> tuple[OperationSymbol, ...]:
    """Return literal symbols in pre-order by a bounded non-recursive walk."""
    symbols: list[OperationSymbol] = []
    pending = [term]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > _MAX_TERM_NODES:
            raise ValidationDefinitionError(
                field="law", reason="term syntax exceeds declared maximum node count"
            )
        if node.symbol is not None:
            symbols.append(node.symbol)
        pending.extend(reversed(node.arguments))
    return tuple(symbols)


def _preflight(structure: object, law: object) -> FiniteSubstitutionDomain:
    """Reject malformed or foreign syntax before evaluating any user operation."""
    if type(structure) is not Structure:
        raise ValidationDefinitionError(
            field="structure", reason="must be an exact Structure"
        )
    if type(law) is not Law:
        raise ValidationDefinitionError(field="law", reason="must be an exact Law")
    for variable in law.variables:
        if _carrier_for_sort(structure, variable.sort) is None:
            raise ValidationDefinitionError(
                field="variables", reason="variable sort is not declared by structure"
            )
    equations = (law.conclusion, *law.hypotheses)
    for equation in equations:
        if _carrier_for_sort(structure, equation.left.sort) is None:
            raise ValidationDefinitionError(
                field="law", reason="equation result sort is not declared by structure"
            )
        for symbol in (*_symbols(equation.left), *_symbols(equation.right)):
            if not any(
                declared == symbol for declared in structure.signature.operations
            ):
                raise ValidationDefinitionError(
                    field="law",
                    reason="law operation symbol is not declared by structure",
                )
    bindings: list[tuple[Sort, FiniteCarrier]] = []
    for variable in law.variables:
        if not any(variable.sort == sort for sort, _carrier in bindings):
            carrier = _carrier_for_sort(structure, variable.sort)
            assert carrier is not None
            bindings.append((variable.sort, carrier))
    try:
        return FiniteSubstitutionDomain.from_bindings(law.variables, bindings)
    except SubstitutionDomainError as error:
        raise ValidationDefinitionError(
            field="variables", reason=error.reason
        ) from None


def _evaluate(
    structure: Structure, term: Term, environment: object
) -> EvaluationResult:
    """Translate unexpected evaluator boundary failure to inert failed evidence."""
    try:
        return evaluate_term(structure, term, environment)
    except Exception:
        return EvaluationResult(
            Failed("term evaluation boundary failed", stage="validation")
        )


def _equal_defined(
    carrier: FiniteCarrier, left: Defined[object], right: Defined[object]
) -> bool | None:
    """Compare defined results through declared-carrier index semantics only."""
    try:
        return carrier.index(left.value) == carrier.index(right.value)
    except CarrierMemberNotFoundError:
        return None


def _witness(
    indices: tuple[int, ...],
    phase: str,
    premise_index: int | None,
    left: EvaluationResult | None,
    right: EvaluationResult | None,
) -> ValidationWitness:
    """Build one inert location/outcome witness without rendering member payloads."""
    return ValidationWitness._create(indices, phase, premise_index, left, right)


def validate_law(structure: Structure, law: Law) -> Proved | Disproved | Inconclusive:
    """Exhaustively check one finite law in substitution-domain lexicographic order.

    An unequal or undefined premise makes its implication vacuous.  Undefined
    conclusions fail under ``strong`` semantics but are skipped under
    ``definedness_conditional``.  Indeterminate or failed results remain
    inconclusive evidence while enumeration continues for a later definite
    counterexample.
    """
    domain = _preflight(structure, law)
    expected = len(domain)
    variable_carrier_sizes = tuple(
        len(
            next(
                carrier
                for sort, carrier in domain.carrier_bindings
                if sort == variable.sort
            )
        )
        for variable in law.variables
    )
    evaluated = premises = conclusions = vacuous = skipped = undecidable = 0
    first_inconclusive: ValidationWitness | None = None
    for substitution in domain:
        evaluated += 1
        indices = substitution.carrier_indices
        assignment_done = False
        for premise_index, premise in enumerate(law.hypotheses):
            premises += 1
            left = _evaluate(structure, premise.left, substitution)
            right = _evaluate(structure, premise.right, substitution)
            if type(left.outcome) is Undefined or type(right.outcome) is Undefined:
                vacuous += 1
                assignment_done = True
                break
            if type(left.outcome) is not Defined or type(right.outcome) is not Defined:
                undecidable += 1
                if first_inconclusive is None:
                    first_inconclusive = _witness(
                        indices, "premise", premise_index, left, right
                    )
                assignment_done = True
                break
            carrier = _carrier_for_sort(structure, premise.left.sort)
            assert carrier is not None
            equal = _equal_defined(carrier, left.outcome, right.outcome)
            if equal is None:
                undecidable += 1
                if first_inconclusive is None:
                    first_inconclusive = _witness(
                        indices, "premise_comparison", premise_index, left, right
                    )
                assignment_done = True
                break
            if not equal:
                vacuous += 1
                assignment_done = True
                break
        if assignment_done:
            continue
        conclusions += 1
        left = _evaluate(structure, law.conclusion.left, substitution)
        right = _evaluate(structure, law.conclusion.right, substitution)
        if type(left.outcome) is Undefined or type(right.outcome) is Undefined:
            if law.partial_semantics == "strong":
                return _report(
                    Disproved,
                    structure,
                    law,
                    expected=expected,
                    evaluated=evaluated,
                    premises=premises,
                    conclusions=conclusions,
                    vacuous=vacuous,
                    skipped=skipped,
                    undecidable=undecidable,
                    variable_carrier_sizes=variable_carrier_sizes,
                    witness=_witness(indices, "conclusion", None, left, right),
                )
            skipped += 1
            continue
        if type(left.outcome) is not Defined or type(right.outcome) is not Defined:
            undecidable += 1
            if first_inconclusive is None:
                first_inconclusive = _witness(indices, "conclusion", None, left, right)
            continue
        carrier = _carrier_for_sort(structure, law.conclusion.left.sort)
        assert carrier is not None
        equal = _equal_defined(carrier, left.outcome, right.outcome)
        if equal is None:
            undecidable += 1
            if first_inconclusive is None:
                first_inconclusive = _witness(
                    indices, "conclusion_comparison", None, left, right
                )
            continue
        if not equal:
            return _report(
                Disproved,
                structure,
                law,
                expected=expected,
                evaluated=evaluated,
                premises=premises,
                conclusions=conclusions,
                vacuous=vacuous,
                skipped=skipped,
                undecidable=undecidable,
                variable_carrier_sizes=variable_carrier_sizes,
                witness=_witness(indices, "conclusion", None, left, right),
            )
    if first_inconclusive is not None:
        return _report(
            Inconclusive,
            structure,
            law,
            expected=expected,
            evaluated=evaluated,
            premises=premises,
            conclusions=conclusions,
            vacuous=vacuous,
            skipped=skipped,
            undecidable=undecidable,
            variable_carrier_sizes=variable_carrier_sizes,
            witness=first_inconclusive,
        )
    return _report(
        Proved,
        structure,
        law,
        expected=expected,
        evaluated=evaluated,
        premises=premises,
        conclusions=conclusions,
        vacuous=vacuous,
        skipped=skipped,
        undecidable=undecidable,
        variable_carrier_sizes=variable_carrier_sizes,
        witness=None,
    )
