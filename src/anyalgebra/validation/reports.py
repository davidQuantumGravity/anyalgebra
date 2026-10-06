"""Immutable schema-ready evidence records for bounded law validation.

This module defines inert evidence records only.  It deliberately does not
encode, decode, import, resolve, or execute anything.  Persistence belongs to a
later milestone.  Composite records reconstruct every nested record so that a
caller cannot mutate retained evidence through ``object.__setattr__`` after a
report has been accepted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from anyalgebra.core.errors import AnyAlgebraError


_MAX_DECLARATIONS = 512
_MAX_ASSIGNMENTS = 65_536
_MAX_WITNESS_COMPONENTS = 65_536
_MAX_TEXT_LENGTH = 512
_PROOF_ROUTES = ("exhaustive_finite", "checked_theorem_reduction")
_ENUMERATIVE_ROUTES = (
    "exhaustive_finite",
    "sampled",
    "bounded_search",
    "witness_minimization",
)
_ROUTES = (*_PROOF_ROUTES, "sampled", "bounded_search", "witness_minimization")
_PARTIAL_SEMANTICS = ("strong", "definedness_conditional")
_HYPOTHESIS_STATUSES = ("declared", "verified", "failed", "unresolved")
_HYPOTHESIS_KINDS = (
    "law_hypothesis",
    "theorem",
    "law_identity",
    "closure",
    "scalar_domain",
    "multilinearity",
    "polarization",
    "multilinearity_or_polarization",
)
_WITNESS_PHASES = (
    "premise",
    "premise_comparison",
    "conclusion",
    "conclusion_comparison",
)
_COUNTEREXAMPLE_SUBTYPES = ("defined_unequal", "strong_undefined")


class ValidationReportDefinitionError(AnyAlgebraError, ValueError):
    """A validation-evidence declaration was malformed."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid validation report {location}: {reason}")


def _text(value: object, field: str, *, index: int | None = None) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_TEXT_LENGTH
    ):
        raise ValidationReportDefinitionError(
            field=field,
            index=index,
            reason="must be a non-empty trimmed built-in str of at most 512 characters",
        )
    return value


def _non_negative(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationReportDefinitionError(
            field=field, reason="must be a non-negative built-in int"
        )
    return value


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValidationReportDefinitionError(
            field=field, reason="must be a positive built-in int"
        )
    return value


def _bounded_snapshot(
    values: Iterable[object],
    *,
    field: str,
    maximum: int,
) -> tuple[object, ...]:
    """Consume a declaration iterable once and stop one pull past its bound."""
    if isinstance(values, str | bytes):
        raise ValidationReportDefinitionError(
            field=field, reason="must be a declaration iterable, not text"
        )
    try:
        iterator = iter(values)
    except Exception:
        raise ValidationReportDefinitionError(
            field=field, reason="could not be snapshotted"
        ) from None
    result: list[object] = []
    index = 0
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise ValidationReportDefinitionError(
                field=field, reason="could not be snapshotted"
            ) from None
        if index == maximum:
            raise ValidationReportDefinitionError(
                field=field, reason=f"exceeds declared maximum {maximum}"
            )
        result.append(value)
        index += 1
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class HypothesisEvidence:
    """One declared premise or one independently checked theorem obligation."""

    name: str
    status: str
    kind: str
    evidence_id: str | None
    algorithm: str | None
    algorithm_version: int | None
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.hypothesis"
    schema_version = 1

    def __init__(
        self,
        name: str,
        status: str = "declared",
        *,
        kind: str = "law_hypothesis",
        evidence_id: str | None = None,
        algorithm: str | None = None,
        algorithm_version: int | None = None,
    ) -> None:
        if type(self) is not HypothesisEvidence:
            raise ValidationReportDefinitionError(
                field="hypothesis", reason="must be an exact HypothesisEvidence"
            )
        checked_name = _text(name, "name")
        if type(status) is not str or status not in _HYPOTHESIS_STATUSES:
            raise ValidationReportDefinitionError(
                field="status", reason="must be a recognized hypothesis status"
            )
        if type(kind) is not str or kind not in _HYPOTHESIS_KINDS:
            raise ValidationReportDefinitionError(
                field="kind", reason="must be a recognized hypothesis kind"
            )
        if status == "declared":
            if (
                evidence_id is not None
                or algorithm is not None
                or algorithm_version is not None
            ):
                raise ValidationReportDefinitionError(
                    field="evidence",
                    reason="declared hypotheses cannot claim checked evidence",
                )
            checked_evidence_id = None
            checked_algorithm = None
            checked_algorithm_version = None
        else:
            checked_evidence_id = _text(evidence_id, "evidence_id")
            checked_algorithm = _text(algorithm, "algorithm")
            checked_algorithm_version = _positive(
                algorithm_version, "algorithm_version"
            )
        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "evidence_id", checked_evidence_id)
        object.__setattr__(self, "algorithm", checked_algorithm)
        object.__setattr__(self, "algorithm_version", checked_algorithm_version)

    def __repr__(self) -> str:
        return (
            "HypothesisEvidence("
            f"name={self.name!r}, status={self.status!r}, kind={self.kind!r})"
        )


def _clone_hypothesis(
    value: object, *, field: str = "hypotheses", index: int | None = None
) -> HypothesisEvidence:
    if type(value) is not HypothesisEvidence:
        raise ValidationReportDefinitionError(
            field=field,
            index=index,
            reason="must contain exact HypothesisEvidence records",
        )
    try:
        name = value.name
        status = value.status
        kind = value.kind
        evidence_id = value.evidence_id
        algorithm = value.algorithm
        algorithm_version = value.algorithm_version
    except Exception:
        raise ValidationReportDefinitionError(
            field=field, index=index, reason="contains a malformed hypothesis record"
        ) from None
    return HypothesisEvidence(
        name,
        status,
        kind=kind,
        evidence_id=evidence_id,
        algorithm=algorithm,
        algorithm_version=algorithm_version,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ActiveBound:
    """One named integer limit active during a validation calculation."""

    name: str
    limit: int
    inclusive: bool
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.bound"
    schema_version = 1

    def __init__(self, name: str, limit: int, *, inclusive: bool = True) -> None:
        if type(self) is not ActiveBound:
            raise ValidationReportDefinitionError(
                field="bound", reason="must be an exact ActiveBound"
            )
        if type(inclusive) is not bool:
            raise ValidationReportDefinitionError(
                field="inclusive", reason="must be a built-in bool"
            )
        object.__setattr__(self, "name", _text(name, "name"))
        object.__setattr__(self, "limit", _non_negative(limit, "limit"))
        object.__setattr__(self, "inclusive", inclusive)

    def __repr__(self) -> str:
        return (
            f"ActiveBound(name={self.name!r}, limit={self.limit}, "
            f"inclusive={self.inclusive})"
        )


def _clone_active_bound(value: object, *, index: int | None = None) -> ActiveBound:
    if type(value) is not ActiveBound:
        raise ValidationReportDefinitionError(
            field="active_bounds",
            index=index,
            reason="must contain exact ActiveBound records",
        )
    try:
        name = value.name
        limit = value.limit
        inclusive = value.inclusive
    except Exception:
        raise ValidationReportDefinitionError(
            field="active_bounds",
            index=index,
            reason="contains a malformed active-bound record",
        ) from None
    return ActiveBound(name, limit, inclusive=inclusive)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ValidationBounds:
    """Route, completeness, finite domains, and every active named limit."""

    route: str
    complete: bool
    assignment_limit: int
    witness_component_limit: int
    domain_cardinalities: tuple[int, ...]
    active_bounds: tuple[ActiveBound, ...]
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.bounds"
    schema_version = 1

    def __init__(
        self,
        route: str,
        complete: bool,
        assignment_limit: int,
        *,
        domain_cardinalities: Iterable[int] = (),
        active_bounds: Iterable[ActiveBound] = (),
        witness_component_limit: int = _MAX_WITNESS_COMPONENTS,
    ) -> None:
        if type(self) is not ValidationBounds:
            raise ValidationReportDefinitionError(
                field="bounds", reason="must be an exact ValidationBounds"
            )
        if type(route) is not str or route not in _ROUTES:
            raise ValidationReportDefinitionError(
                field="route", reason="must be a recognized validation route"
            )
        if type(complete) is not bool:
            raise ValidationReportDefinitionError(
                field="complete", reason="must be a built-in bool"
            )
        if route in ("sampled", "bounded_search") and complete:
            raise ValidationReportDefinitionError(
                field="complete", reason=f"{route} cannot claim complete coverage"
            )
        checked_assignment_limit = _positive(assignment_limit, "assignment_limit")
        if checked_assignment_limit > _MAX_ASSIGNMENTS:
            raise ValidationReportDefinitionError(
                field="assignment_limit", reason="exceeds declared maximum 65536"
            )
        checked_witness_limit = _positive(
            witness_component_limit, "witness_component_limit"
        )
        if checked_witness_limit > _MAX_WITNESS_COMPONENTS:
            raise ValidationReportDefinitionError(
                field="witness_component_limit",
                reason="exceeds declared maximum 65536",
            )
        raw_cardinalities = _bounded_snapshot(
            domain_cardinalities,
            field="domain_cardinalities",
            maximum=_MAX_DECLARATIONS,
        )
        cardinalities = tuple(
            _positive(value, "domain_cardinalities") for value in raw_cardinalities
        )
        domain_size = 1
        for cardinality in cardinalities:
            domain_size *= cardinality
            if domain_size > checked_assignment_limit:
                raise ValidationReportDefinitionError(
                    field="domain_cardinalities",
                    reason="product exceeds assignment_limit",
                )
        if route == "checked_theorem_reduction" and cardinalities:
            raise ValidationReportDefinitionError(
                field="domain_cardinalities",
                reason="theorem reduction does not enumerate a finite domain",
            )

        checked_bounds: list[ActiveBound] = []
        names: set[str] = set()
        if isinstance(active_bounds, str | bytes):
            raise ValidationReportDefinitionError(
                field="active_bounds", reason="must be a declaration iterable, not text"
            )
        try:
            bound_iterator = iter(active_bounds)
        except Exception:
            raise ValidationReportDefinitionError(
                field="active_bounds", reason="could not be snapshotted"
            ) from None
        index = 0
        while True:
            try:
                bound = next(bound_iterator)
            except StopIteration:
                break
            except Exception:
                raise ValidationReportDefinitionError(
                    field="active_bounds", reason="could not be snapshotted"
                ) from None
            if index == _MAX_DECLARATIONS:
                raise ValidationReportDefinitionError(
                    field="active_bounds",
                    reason="exceeds declared maximum 512",
                )
            checked = _clone_active_bound(bound, index=index)
            if checked.name in names:
                raise ValidationReportDefinitionError(
                    field="active_bounds",
                    index=index,
                    reason="contains a duplicate bound name",
                )
            names.add(checked.name)
            checked_bounds.append(checked)
            index += 1

        object.__setattr__(self, "route", route)
        object.__setattr__(self, "complete", complete)
        object.__setattr__(self, "assignment_limit", checked_assignment_limit)
        object.__setattr__(self, "witness_component_limit", checked_witness_limit)
        object.__setattr__(self, "domain_cardinalities", cardinalities)
        object.__setattr__(self, "active_bounds", tuple(checked_bounds))

    def __repr__(self) -> str:
        return (
            "ValidationBounds("
            f"route={self.route!r}, complete={self.complete}, "
            f"assignment_limit={self.assignment_limit}, "
            f"domain_rank={len(self.domain_cardinalities)}, "
            f"active_bound_count={len(self.active_bounds)})"
        )


def _clone_bounds(value: object) -> ValidationBounds:
    if type(value) is not ValidationBounds:
        raise ValidationReportDefinitionError(
            field="bounds", reason="must be an exact ValidationBounds"
        )
    try:
        route = value.route
        complete = value.complete
        assignment_limit = value.assignment_limit
        domain_cardinalities = value.domain_cardinalities
        active_bounds = value.active_bounds
        witness_component_limit = value.witness_component_limit
    except Exception:
        raise ValidationReportDefinitionError(
            field="bounds", reason="contains a malformed ValidationBounds record"
        ) from None
    if type(domain_cardinalities) is not tuple or type(active_bounds) is not tuple:
        raise ValidationReportDefinitionError(
            field="bounds", reason="contains malformed nested declaration storage"
        )
    return ValidationBounds(
        route,
        complete,
        assignment_limit,
        domain_cardinalities=domain_cardinalities,
        active_bounds=active_bounds,
        witness_component_limit=witness_component_limit,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ValidationCounts:
    """A coherent terminal partition of one bounded validation run."""

    assignments_expected: int
    assignments_evaluated: int
    hypotheses_evaluated: int
    hypothesis_rejected: int
    conclusions_evaluated: int
    compared: int
    vacuous_assignments: int
    skipped_assignments: int
    undecidable_assignments: int
    undefined_left: int
    undefined_right: int
    undefined_both: int
    conclusion_false: int
    evaluation_failed: int
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.counts"
    schema_version = 1

    def __init__(
        self,
        assignments_expected: int,
        assignments_evaluated: int,
        *,
        hypotheses_evaluated: int = 0,
        hypothesis_rejected: int = 0,
        conclusions_evaluated: int = 0,
        compared: int = 0,
        vacuous_assignments: int = 0,
        skipped_assignments: int = 0,
        undecidable_assignments: int = 0,
        undefined_left: int = 0,
        undefined_right: int = 0,
        undefined_both: int = 0,
        conclusion_false: int = 0,
        evaluation_failed: int = 0,
    ) -> None:
        if type(self) is not ValidationCounts:
            raise ValidationReportDefinitionError(
                field="counts", reason="must be an exact ValidationCounts"
            )
        values = {
            name: _non_negative(value, name)
            for name, value in (
                ("assignments_expected", assignments_expected),
                ("assignments_evaluated", assignments_evaluated),
                ("hypotheses_evaluated", hypotheses_evaluated),
                ("hypothesis_rejected", hypothesis_rejected),
                ("conclusions_evaluated", conclusions_evaluated),
                ("compared", compared),
                ("vacuous_assignments", vacuous_assignments),
                ("skipped_assignments", skipped_assignments),
                ("undecidable_assignments", undecidable_assignments),
                ("undefined_left", undefined_left),
                ("undefined_right", undefined_right),
                ("undefined_both", undefined_both),
                ("conclusion_false", conclusion_false),
                ("evaluation_failed", evaluation_failed),
            )
        }
        expected = values["assignments_expected"]
        evaluated = values["assignments_evaluated"]
        vacuous = values["vacuous_assignments"]
        undecidable = values["undecidable_assignments"]
        compared = values["compared"]
        conclusions = values["conclusions_evaluated"]
        undefined_total = (
            values["undefined_left"]
            + values["undefined_right"]
            + values["undefined_both"]
        )
        if expected > _MAX_ASSIGNMENTS:
            raise ValidationReportDefinitionError(
                field="assignments_expected", reason="exceeds declared maximum 65536"
            )
        if evaluated > expected:
            raise ValidationReportDefinitionError(
                field="assignments_evaluated",
                reason="must not exceed assignments_expected",
            )
        if undecidable > evaluated:
            raise ValidationReportDefinitionError(
                field="undecidable_assignments",
                reason="must not exceed assignments_evaluated",
            )
        if vacuous + undecidable + compared + undefined_total != evaluated:
            raise ValidationReportDefinitionError(
                field="terminal_partition",
                reason=(
                    "vacuous, undecidable, compared, and undefined outcomes "
                    "must partition assignments_evaluated"
                ),
            )
        if conclusions > evaluated - vacuous:
            raise ValidationReportDefinitionError(
                field="conclusions_evaluated",
                reason="exceeds non-vacuous evaluated assignments",
            )
        if compared + undefined_total > conclusions:
            raise ValidationReportDefinitionError(
                field="conclusion_outcomes",
                reason="compared and undefined outcomes exceed conclusions_evaluated",
            )
        if values["hypothesis_rejected"] > vacuous:
            raise ValidationReportDefinitionError(
                field="hypothesis_rejected",
                reason="must not exceed vacuous_assignments",
            )
        if values["hypothesis_rejected"] > values["hypotheses_evaluated"]:
            raise ValidationReportDefinitionError(
                field="hypothesis_rejected",
                reason="must not exceed hypotheses_evaluated",
            )
        if vacuous > values["hypotheses_evaluated"]:
            raise ValidationReportDefinitionError(
                field="vacuous_assignments",
                reason="must not exceed hypotheses_evaluated",
            )
        if values["skipped_assignments"] > undefined_total:
            raise ValidationReportDefinitionError(
                field="skipped_assignments",
                reason="must not exceed undefined conclusion outcomes",
            )
        if values["conclusion_false"] > compared + undefined_total:
            raise ValidationReportDefinitionError(
                field="conclusion_false",
                reason="requires a compared or strong-undefined conclusion",
            )
        if values["evaluation_failed"] > undecidable:
            raise ValidationReportDefinitionError(
                field="evaluation_failed",
                reason="must not exceed undecidable_assignments",
            )
        if evaluated == 0 and any(
            value
            for name, value in values.items()
            if name not in ("assignments_expected", "assignments_evaluated")
        ):
            raise ValidationReportDefinitionError(
                field="counts",
                reason="nonzero coverage requires an evaluated assignment",
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def undefined_total(self) -> int:
        """Return the disjoint total of undefined conclusion subtypes."""
        return self.undefined_left + self.undefined_right + self.undefined_both

    def __repr__(self) -> str:
        return (
            "ValidationCounts("
            f"assignments_expected={self.assignments_expected}, "
            f"assignments_evaluated={self.assignments_evaluated}, "
            f"compared={self.compared}, conclusion_false={self.conclusion_false})"
        )


def _clone_counts(value: object) -> ValidationCounts:
    if type(value) is not ValidationCounts:
        raise ValidationReportDefinitionError(
            field="counts", reason="must be an exact ValidationCounts"
        )
    names = (
        "assignments_expected",
        "assignments_evaluated",
        "hypotheses_evaluated",
        "hypothesis_rejected",
        "conclusions_evaluated",
        "compared",
        "vacuous_assignments",
        "skipped_assignments",
        "undecidable_assignments",
        "undefined_left",
        "undefined_right",
        "undefined_both",
        "conclusion_false",
        "evaluation_failed",
    )
    try:
        values = tuple(getattr(value, name) for name in names)
    except Exception:
        raise ValidationReportDefinitionError(
            field="counts", reason="contains a malformed ValidationCounts record"
        ) from None
    return ValidationCounts(
        values[0],
        values[1],
        hypotheses_evaluated=values[2],
        hypothesis_rejected=values[3],
        conclusions_evaluated=values[4],
        compared=values[5],
        vacuous_assignments=values[6],
        skipped_assignments=values[7],
        undecidable_assignments=values[8],
        undefined_left=values[9],
        undefined_right=values[10],
        undefined_both=values[11],
        conclusion_false=values[12],
        evaluation_failed=values[13],
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DeterministicWitness:
    """Safe coordinates and explicit prefix evidence for a validation outcome."""

    kind: str
    order_name: str
    order_algorithm: str
    order_version: int
    minimality: str
    assignment_index: int
    carrier_indices: tuple[int, ...]
    phase: str
    premise_index: int | None
    counterexample_subtype: str | None
    order_size: int
    declared_order: tuple[int, ...]
    evaluated_indices: tuple[int, ...]
    earlier_resolved: bool
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.witness"
    schema_version = 1

    def __init__(
        self,
        kind: str,
        order_name: str,
        order_algorithm: str,
        order_version: int,
        minimality: str,
        assignment_index: int,
        carrier_indices: Iterable[int],
        phase: str,
        premise_index: int | None = None,
        *,
        counterexample_subtype: str | None = None,
        order_size: int,
        declared_order: Iterable[int],
        evaluated_indices: Iterable[int],
        earlier_resolved: bool,
    ) -> None:
        if type(self) is not DeterministicWitness:
            raise ValidationReportDefinitionError(
                field="witness", reason="must be an exact DeterministicWitness"
            )
        if type(kind) is not str or kind not in ("counterexample", "evaluation_gap"):
            raise ValidationReportDefinitionError(
                field="kind", reason="must be 'counterexample' or 'evaluation_gap'"
            )
        if type(phase) is not str or phase not in _WITNESS_PHASES:
            raise ValidationReportDefinitionError(
                field="phase", reason="must be a recognized validation phase"
            )
        if type(minimality) is not str or minimality not in (
            "least_under_declared_order",
            "not_applicable",
        ):
            raise ValidationReportDefinitionError(
                field="minimality", reason="must be a recognized minimality policy"
            )
        if type(earlier_resolved) is not bool:
            raise ValidationReportDefinitionError(
                field="earlier_resolved", reason="must be a built-in bool"
            )
        checked_order_size = _positive(order_size, "order_size")
        if checked_order_size > _MAX_ASSIGNMENTS:
            raise ValidationReportDefinitionError(
                field="order_size", reason="exceeds declared maximum 65536"
            )
        raw_declared_order = _bounded_snapshot(
            declared_order,
            field="declared_order",
            maximum=_MAX_ASSIGNMENTS,
        )
        checked_declared_order = tuple(
            _non_negative(value, "declared_order") for value in raw_declared_order
        )
        if len(checked_declared_order) != checked_order_size or set(
            checked_declared_order
        ) != set(range(checked_order_size)):
            raise ValidationReportDefinitionError(
                field="declared_order",
                reason="must be a full permutation of range(order_size)",
            )
        raw_evaluated_indices = _bounded_snapshot(
            evaluated_indices,
            field="evaluated_indices",
            maximum=_MAX_ASSIGNMENTS,
        )
        checked_evaluated_indices = tuple(
            _non_negative(value, "evaluated_indices") for value in raw_evaluated_indices
        )
        if not checked_evaluated_indices:
            raise ValidationReportDefinitionError(
                field="evaluated_indices",
                reason="must contain at least the witnessed assignment",
            )
        if any(index >= checked_order_size for index in checked_evaluated_indices):
            raise ValidationReportDefinitionError(
                field="evaluated_indices", reason="must stay inside order_size"
            )
        if len(set(checked_evaluated_indices)) != len(checked_evaluated_indices):
            raise ValidationReportDefinitionError(
                field="evaluated_indices", reason="must not contain duplicates"
            )
        checked_assignment_index = _non_negative(assignment_index, "assignment_index")
        if checked_assignment_index >= checked_order_size:
            raise ValidationReportDefinitionError(
                field="assignment_index", reason="must be inside order_size"
            )
        if checked_evaluated_indices[-1] != checked_assignment_index:
            raise ValidationReportDefinitionError(
                field="evaluated_indices",
                reason="the witnessed assignment must be the final evaluated index",
            )
        if (
            checked_evaluated_indices
            != checked_declared_order[: len(checked_evaluated_indices)]
        ):
            raise ValidationReportDefinitionError(
                field="evaluated_indices",
                reason="must be the leading prefix of declared_order",
            )

        if kind == "counterexample":
            if phase != "conclusion" or premise_index is not None:
                raise ValidationReportDefinitionError(
                    field="phase",
                    reason="counterexamples must be conclusion witnesses",
                )
            if minimality != "least_under_declared_order":
                raise ValidationReportDefinitionError(
                    field="minimality",
                    reason="counterexamples require declared-order leastness",
                )
            if (
                type(counterexample_subtype) is not str
                or counterexample_subtype not in _COUNTEREXAMPLE_SUBTYPES
            ):
                raise ValidationReportDefinitionError(
                    field="counterexample_subtype",
                    reason="must be 'defined_unequal' or 'strong_undefined'",
                )
            if not earlier_resolved:
                raise ValidationReportDefinitionError(
                    field="earlier_resolved",
                    reason="leastness requires every earlier candidate to be resolved",
                )
        else:
            if minimality != "not_applicable":
                raise ValidationReportDefinitionError(
                    field="minimality",
                    reason="evaluation gaps have no minimality claim",
                )
            if counterexample_subtype is not None:
                raise ValidationReportDefinitionError(
                    field="counterexample_subtype",
                    reason="evaluation gaps cannot claim a counterexample subtype",
                )
        if phase.startswith("premise"):
            if type(premise_index) is not int or premise_index < 0:
                raise ValidationReportDefinitionError(
                    field="premise_index", reason="must be a non-negative built-in int"
                )
        elif premise_index is not None:
            raise ValidationReportDefinitionError(
                field="premise_index", reason="must be None for a conclusion phase"
            )

        raw_indices = _bounded_snapshot(
            carrier_indices,
            field="carrier_indices",
            maximum=_MAX_WITNESS_COMPONENTS,
        )
        indices = tuple(
            _non_negative(value, "carrier_indices") for value in raw_indices
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "order_name", _text(order_name, "order_name"))
        object.__setattr__(
            self, "order_algorithm", _text(order_algorithm, "order_algorithm")
        )
        object.__setattr__(
            self, "order_version", _positive(order_version, "order_version")
        )
        object.__setattr__(self, "minimality", minimality)
        object.__setattr__(self, "assignment_index", checked_assignment_index)
        object.__setattr__(self, "carrier_indices", indices)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "premise_index", premise_index)
        object.__setattr__(self, "counterexample_subtype", counterexample_subtype)
        object.__setattr__(self, "order_size", checked_order_size)
        object.__setattr__(self, "declared_order", checked_declared_order)
        object.__setattr__(self, "evaluated_indices", checked_evaluated_indices)
        object.__setattr__(self, "earlier_resolved", earlier_resolved)

    def __repr__(self) -> str:
        return (
            "DeterministicWitness("
            f"kind={self.kind!r}, assignment_index={self.assignment_index}, "
            f"phase={self.phase!r}, evaluated_count={len(self.evaluated_indices)}, "
            f"component_count={len(self.carrier_indices)})"
        )


def _clone_witness(value: object) -> DeterministicWitness:
    if type(value) is not DeterministicWitness:
        raise ValidationReportDefinitionError(
            field="witness", reason="must be an exact DeterministicWitness"
        )
    try:
        kind = value.kind
        order_name = value.order_name
        order_algorithm = value.order_algorithm
        order_version = value.order_version
        minimality = value.minimality
        assignment_index = value.assignment_index
        carrier_indices = value.carrier_indices
        phase = value.phase
        premise_index = value.premise_index
        counterexample_subtype = value.counterexample_subtype
        order_size = value.order_size
        declared_order = value.declared_order
        evaluated_indices = value.evaluated_indices
        earlier_resolved = value.earlier_resolved
    except Exception:
        raise ValidationReportDefinitionError(
            field="witness", reason="contains a malformed DeterministicWitness record"
        ) from None
    if (
        type(carrier_indices) is not tuple
        or type(declared_order) is not tuple
        or type(evaluated_indices) is not tuple
    ):
        raise ValidationReportDefinitionError(
            field="witness", reason="contains malformed nested declaration storage"
        )
    return DeterministicWitness(
        kind,
        order_name,
        order_algorithm,
        order_version,
        minimality,
        assignment_index,
        carrier_indices,
        phase,
        premise_index,
        counterexample_subtype=counterexample_subtype,
        order_size=order_size,
        declared_order=declared_order,
        evaluated_indices=evaluated_indices,
        earlier_resolved=earlier_resolved,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ValidationReport:
    """Common schema-ready evidence fields for a validation outcome."""

    hypotheses: tuple[HypothesisEvidence, ...]
    counts: ValidationCounts
    bounds: ValidationBounds
    algorithm: str
    algorithm_version: int
    partial_semantics: str
    witness: DeterministicWitness | None
    __hash__ = None  # type: ignore[assignment]

    schema_tag = "anyalgebra.validation.report"
    schema_version = 1

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValidationReportDefinitionError(
            field="report", reason="must use a concrete outcome type"
        )

    @property
    def status(self) -> str:
        if type(self) is Proved:
            return "proved"
        if type(self) is Disproved:
            return "disproved"
        if type(self) is Inconclusive:
            return "inconclusive"
        raise ValidationReportDefinitionError(
            field="report", reason="has an unknown concrete outcome type"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status={self.status!r}, "
            f"hypothesis_count={len(self.hypotheses)}, "
            f"assignments_evaluated={self.counts.assignments_evaluated}, "
            f"partial_semantics={self.partial_semantics!r}, "
            f"has_witness={self.witness is not None})"
        )


_ReportFields: TypeAlias = tuple[
    tuple[HypothesisEvidence, ...],
    ValidationCounts,
    ValidationBounds,
    str,
    int,
    str,
    DeterministicWitness | None,
]


def _domain_size(cardinalities: tuple[int, ...]) -> int:
    size = 1
    for cardinality in cardinalities:
        size *= cardinality
    return size


def _flat_assignment_index(
    carrier_indices: tuple[int, ...], cardinalities: tuple[int, ...]
) -> int:
    result = 0
    for index, cardinality in zip(carrier_indices, cardinalities, strict=True):
        result = result * cardinality + index
    return result


def _validate_partial_counts(counts: ValidationCounts, partial_semantics: str) -> None:
    undefined_total = counts.undefined_total
    if partial_semantics == "strong":
        if counts.skipped_assignments != 0:
            raise ValidationReportDefinitionError(
                field="skipped_assignments",
                reason="strong partial semantics cannot skip undefined conclusions",
            )
        if counts.conclusion_false < undefined_total:
            raise ValidationReportDefinitionError(
                field="conclusion_false",
                reason="strong undefined conclusions must contribute to false coverage",
            )
    else:
        if counts.skipped_assignments != undefined_total:
            raise ValidationReportDefinitionError(
                field="skipped_assignments",
                reason=(
                    "definedness-conditional semantics must skip every undefined "
                    "conclusion"
                ),
            )
        if counts.conclusion_false > counts.compared:
            raise ValidationReportDefinitionError(
                field="conclusion_false",
                reason="definedness-conditional false coverage requires comparison",
            )


def _report_fields(
    hypotheses: Iterable[HypothesisEvidence],
    counts: ValidationCounts,
    bounds: ValidationBounds,
    algorithm: str,
    algorithm_version: int,
    partial_semantics: str,
    witness: DeterministicWitness | None,
) -> _ReportFields:
    checked_counts = _clone_counts(counts)
    checked_bounds = _clone_bounds(bounds)
    if (
        type(partial_semantics) is not str
        or partial_semantics not in _PARTIAL_SEMANTICS
    ):
        raise ValidationReportDefinitionError(
            field="partial_semantics",
            reason="must be 'strong' or 'definedness_conditional'",
        )
    if checked_counts.assignments_expected > checked_bounds.assignment_limit:
        raise ValidationReportDefinitionError(
            field="counts", reason="assignments_expected exceeds assignment_limit"
        )
    if checked_bounds.route == "checked_theorem_reduction":
        if (
            checked_counts.assignments_expected != 0
            or checked_counts.assignments_evaluated != 0
        ):
            raise ValidationReportDefinitionError(
                field="counts", reason="theorem-reduction coverage counts must be zero"
            )
    else:
        expected_domain_size = _domain_size(checked_bounds.domain_cardinalities)
        if checked_counts.assignments_expected != expected_domain_size:
            raise ValidationReportDefinitionError(
                field="assignments_expected",
                reason="must equal the finite mixed-radix domain product",
            )

    checked_witness = None if witness is None else _clone_witness(witness)
    checked_hypotheses: list[HypothesisEvidence] = []
    names: set[str] = set()
    if isinstance(hypotheses, str | bytes):
        raise ValidationReportDefinitionError(
            field="hypotheses", reason="must be a declaration iterable, not text"
        )
    try:
        hypothesis_iterator = iter(hypotheses)
    except Exception:
        raise ValidationReportDefinitionError(
            field="hypotheses", reason="could not be snapshotted"
        ) from None
    index = 0
    while True:
        try:
            hypothesis = next(hypothesis_iterator)
        except StopIteration:
            break
        except Exception:
            raise ValidationReportDefinitionError(
                field="hypotheses", reason="could not be snapshotted"
            ) from None
        if index == _MAX_DECLARATIONS:
            raise ValidationReportDefinitionError(
                field="hypotheses", reason="exceeds declared maximum 512"
            )
        checked = _clone_hypothesis(hypothesis, index=index)
        if checked.name in names:
            raise ValidationReportDefinitionError(
                field="hypotheses",
                index=index,
                reason="contains a duplicate hypothesis name",
            )
        names.add(checked.name)
        checked_hypotheses.append(checked)
        index += 1
    if checked_bounds.route in _ENUMERATIVE_ROUTES and any(
        hypothesis.kind != "law_hypothesis" for hypothesis in checked_hypotheses
    ):
        raise ValidationReportDefinitionError(
            field="hypotheses",
            reason="enumerative routes accept only law-hypothesis evidence",
        )
    law_hypothesis_count = sum(
        hypothesis.kind == "law_hypothesis" for hypothesis in checked_hypotheses
    )
    maximum_hypothesis_evaluations = (
        checked_counts.assignments_evaluated * law_hypothesis_count
    )
    if checked_counts.hypotheses_evaluated > maximum_hypothesis_evaluations:
        raise ValidationReportDefinitionError(
            field="hypotheses_evaluated",
            reason="exceeds evaluated assignments times declared law hypotheses",
        )
    if checked_witness is not None and checked_witness.phase.startswith("premise"):
        if (
            checked_witness.premise_index is None
            or checked_witness.premise_index >= len(checked_hypotheses)
        ):
            raise ValidationReportDefinitionError(
                field="premise_index",
                reason="must identify a declared hypothesis",
            )
        if (
            checked_witness.premise_index is not None
            and checked_counts.hypotheses_evaluated <= checked_witness.premise_index
        ):
            raise ValidationReportDefinitionError(
                field="hypotheses_evaluated",
                reason="premise witness requires evaluation through its premise index",
            )
    if law_hypothesis_count > 0:
        minimum_hypothesis_evaluations = (
            checked_counts.assignments_evaluated
            + checked_counts.conclusions_evaluated * (law_hypothesis_count - 1)
        )
        if (
            checked_witness is not None
            and checked_witness.kind == "evaluation_gap"
            and checked_witness.phase.startswith("premise")
            and checked_witness.premise_index is not None
        ):
            minimum_hypothesis_evaluations += checked_witness.premise_index
        if checked_counts.hypotheses_evaluated < minimum_hypothesis_evaluations:
            raise ValidationReportDefinitionError(
                field="hypotheses_evaluated",
                reason="is below the ordered premise-evaluation minimum",
            )

    if checked_witness is not None:
        if checked_counts.assignments_evaluated == 0:
            raise ValidationReportDefinitionError(
                field="witness", reason="requires an evaluated assignment"
            )
        if checked_witness.order_size != checked_counts.assignments_expected:
            raise ValidationReportDefinitionError(
                field="witness", reason="order_size must equal assignments_expected"
            )
        if (
            len(checked_witness.evaluated_indices)
            > checked_counts.assignments_evaluated
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="evaluated indices exceed assignments_evaluated",
            )
        if len(checked_witness.carrier_indices) != len(
            checked_bounds.domain_cardinalities
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="carrier-index arity must match domain cardinalities",
            )
        if any(
            index >= cardinality
            for index, cardinality in zip(
                checked_witness.carrier_indices,
                checked_bounds.domain_cardinalities,
                strict=True,
            )
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="carrier index is outside its domain cardinality",
            )
        if (
            _flat_assignment_index(
                checked_witness.carrier_indices,
                checked_bounds.domain_cardinalities,
            )
            != checked_witness.assignment_index
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="carrier coordinates do not match the flat assignment index",
            )
        if (
            len(checked_witness.carrier_indices)
            > checked_bounds.witness_component_limit
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="component count exceeds witness_component_limit",
            )

    _validate_partial_counts(checked_counts, partial_semantics)
    return (
        tuple(checked_hypotheses),
        checked_counts,
        checked_bounds,
        _text(algorithm, "algorithm"),
        _positive(algorithm_version, "algorithm_version"),
        partial_semantics,
        checked_witness,
    )


def _set_common(report: ValidationReport, fields: _ReportFields) -> None:
    for name, value in zip(
        (
            "hypotheses",
            "counts",
            "bounds",
            "algorithm",
            "algorithm_version",
            "partial_semantics",
            "witness",
        ),
        fields,
        strict=True,
    ):
        object.__setattr__(report, name, value)


def _require_checked_theorem_hypotheses(
    hypotheses: tuple[HypothesisEvidence, ...],
) -> None:
    if not hypotheses or any(item.status != "verified" for item in hypotheses):
        raise ValidationReportDefinitionError(
            field="hypotheses",
            reason="theorem reduction requires only nonempty verified evidence",
        )
    evidence_ids = tuple(item.evidence_id for item in hypotheses)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValidationReportDefinitionError(
            field="hypotheses",
            reason="theorem obligations require distinct evidence identifiers",
        )
    kinds = {item.kind for item in hypotheses}
    missing = {
        "theorem",
        "law_identity",
        "closure",
        "scalar_domain",
    } - kinds
    if missing:
        raise ValidationReportDefinitionError(
            field="hypotheses",
            reason="theorem reduction is missing a required verified obligation kind",
        )
    if not kinds.intersection(
        {"multilinearity", "polarization", "multilinearity_or_polarization"}
    ):
        raise ValidationReportDefinitionError(
            field="hypotheses",
            reason="theorem reduction requires multilinearity or polarization evidence",
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Proved(ValidationReport):
    """Complete exhaustive evidence or a checked theorem-reduction proof."""

    proof_route: str

    def __init__(
        self,
        hypotheses: Iterable[HypothesisEvidence],
        counts: ValidationCounts,
        bounds: ValidationBounds,
        algorithm: str,
        algorithm_version: int,
        *,
        proof_route: str,
        partial_semantics: str = "strong",
    ) -> None:
        if type(self) is not Proved:
            raise ValidationReportDefinitionError(
                field="report", reason="must be an exact Proved"
            )
        fields = _report_fields(
            hypotheses,
            counts,
            bounds,
            algorithm,
            algorithm_version,
            partial_semantics,
            None,
        )
        checked_counts = fields[1]
        checked_bounds = fields[2]
        checked_hypotheses = fields[0]
        if type(proof_route) is not str or proof_route not in _PROOF_ROUTES:
            raise ValidationReportDefinitionError(
                field="proof_route", reason="must be a recognized complete proof route"
            )
        if proof_route != checked_bounds.route or not checked_bounds.complete:
            raise ValidationReportDefinitionError(
                field="bounds",
                reason="proved evidence requires its complete proof route",
            )
        if (
            checked_counts.undecidable_assignments
            or checked_counts.evaluation_failed
            or checked_counts.conclusion_false
        ):
            raise ValidationReportDefinitionError(
                field="counts",
                reason=(
                    "proved evidence cannot retain false, failed, "
                    "or unresolved coverage"
                ),
            )
        if any(
            hypothesis.status in ("failed", "unresolved")
            for hypothesis in checked_hypotheses
        ):
            raise ValidationReportDefinitionError(
                field="hypotheses",
                reason="proved evidence cannot retain failed or unresolved hypotheses",
            )
        if (
            partial_semantics == "definedness_conditional"
            and checked_counts.compared == 0
            and checked_counts.vacuous_assignments
            != checked_counts.assignments_evaluated
        ):
            raise ValidationReportDefinitionError(
                field="counts",
                reason="definedness-conditional proof requires a compared conclusion",
            )
        if proof_route == "exhaustive_finite":
            if (
                checked_counts.assignments_expected == 0
                or checked_counts.assignments_expected
                != checked_counts.assignments_evaluated
            ):
                raise ValidationReportDefinitionError(
                    field="counts",
                    reason="exhaustive proof requires complete non-empty coverage",
                )
            # The exact terminal partition plus zero unresolved/false coverage
            # already entails vacuous + conclusion coverage == evaluated here.
        else:
            _require_checked_theorem_hypotheses(checked_hypotheses)
        _set_common(self, fields)
        object.__setattr__(self, "proof_route", proof_route)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Disproved(ValidationReport):
    """Negative evidence with a deterministic declared-order least witness."""

    def __init__(
        self,
        hypotheses: Iterable[HypothesisEvidence],
        counts: ValidationCounts,
        bounds: ValidationBounds,
        algorithm: str,
        algorithm_version: int,
        witness: DeterministicWitness,
        *,
        partial_semantics: str = "strong",
    ) -> None:
        if type(self) is not Disproved:
            raise ValidationReportDefinitionError(
                field="report", reason="must be an exact Disproved"
            )
        fields = _report_fields(
            hypotheses,
            counts,
            bounds,
            algorithm,
            algorithm_version,
            partial_semantics,
            witness,
        )
        checked_counts = fields[1]
        checked_bounds = fields[2]
        checked_witness = fields[6]
        if checked_bounds.route not in ("exhaustive_finite", "witness_minimization"):
            raise ValidationReportDefinitionError(
                field="bounds",
                reason=(
                    "least disproof requires exhaustive_finite or "
                    "witness_minimization evidence"
                ),
            )
        if (
            checked_witness is None
            or checked_witness.kind != "counterexample"
            or checked_witness.phase != "conclusion"
            or checked_witness.minimality != "least_under_declared_order"
            or not checked_witness.earlier_resolved
            or len(checked_witness.evaluated_indices)
            != checked_counts.assignments_evaluated
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason="disproved evidence requires a complete resolved least prefix",
            )
        if (
            checked_counts.assignments_evaluated == 0
            or checked_counts.conclusion_false != 1
            or checked_counts.undecidable_assignments != 0
        ):
            raise ValidationReportDefinitionError(
                field="counts",
                reason=(
                    "resolved least disproof requires exactly its witnessed false "
                    "outcome and no undecidable earlier assignment"
                ),
            )
        if checked_witness.counterexample_subtype == "strong_undefined":
            if partial_semantics != "strong" or checked_counts.undefined_total == 0:
                raise ValidationReportDefinitionError(
                    field="witness",
                    reason=(
                        "strong_undefined requires strong semantics and undefined "
                        "false coverage"
                    ),
                )
        elif checked_counts.compared == 0 or (
            partial_semantics == "strong" and checked_counts.undefined_total
        ):
            raise ValidationReportDefinitionError(
                field="witness",
                reason=("defined_unequal requires unambiguous compared false coverage"),
            )
        _set_common(self, fields)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Inconclusive(ValidationReport):
    """Explicitly incomplete or unresolved evidence that cannot prove a law."""

    reasons: tuple[str, ...]

    def __init__(
        self,
        hypotheses: Iterable[HypothesisEvidence],
        counts: ValidationCounts,
        bounds: ValidationBounds,
        algorithm: str,
        algorithm_version: int,
        *,
        reasons: Iterable[str],
        witness: DeterministicWitness | None = None,
        partial_semantics: str = "strong",
    ) -> None:
        if type(self) is not Inconclusive:
            raise ValidationReportDefinitionError(
                field="report", reason="must be an exact Inconclusive"
            )
        fields = _report_fields(
            hypotheses,
            counts,
            bounds,
            algorithm,
            algorithm_version,
            partial_semantics,
            witness,
        )
        raw_reasons = _bounded_snapshot(
            reasons, field="reasons", maximum=_MAX_DECLARATIONS
        )
        checked_reasons: list[str] = []
        seen: set[str] = set()
        for index, reason in enumerate(raw_reasons):
            value = _text(reason, "reasons", index=index)
            if value in seen:
                raise ValidationReportDefinitionError(
                    field="reasons", index=index, reason="contains a duplicate reason"
                )
            seen.add(value)
            checked_reasons.append(value)
        if not checked_reasons:
            raise ValidationReportDefinitionError(
                field="reasons", reason="inconclusive evidence requires a reason"
            )
        checked_counts = fields[1]
        checked_bounds = fields[2]
        checked_witness = fields[6]
        if checked_counts.conclusion_false:
            raise ValidationReportDefinitionError(
                field="counts", reason="false coverage must be represented as Disproved"
            )
        if checked_witness is not None and checked_witness.kind != "evaluation_gap":
            raise ValidationReportDefinitionError(
                field="witness",
                reason="inconclusive evidence may retain only an evaluation gap",
            )
        if checked_witness is not None and checked_counts.undecidable_assignments == 0:
            raise ValidationReportDefinitionError(
                field="counts",
                reason="evaluation gap requires an undecidable assignment",
            )
        if (
            checked_bounds.complete
            and checked_counts.assignments_expected
            == checked_counts.assignments_evaluated
            and checked_counts.undecidable_assignments == 0
            and checked_counts.evaluation_failed == 0
            and checked_counts.compared > 0
            and checked_witness is None
            and not any(
                hypothesis.status in ("failed", "unresolved")
                for hypothesis in fields[0]
            )
        ):
            raise ValidationReportDefinitionError(
                field="bounds",
                reason="clean complete evidence cannot be inconclusive",
            )
        _set_common(self, fields)
        object.__setattr__(self, "reasons", tuple(checked_reasons))
