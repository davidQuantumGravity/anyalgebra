"""Declared-order finite counterexample minimization without heuristic shrinking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.structures.evaluate import EvaluationResult
from anyalgebra.structures.laws import Law
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.structure import Structure
from anyalgebra.validation.domains import FiniteSubstitutionDomain
from anyalgebra.validation.sampling import (
    SampledDisproved,
    SamplingOptions,
    _sample_indices,
)
from anyalgebra.validation.validate import (
    Disproved,
    ValidationDefinitionError,
    ValidationWitness,
    _carrier_for_sort,
    _equal_defined,
    _evaluate,
    _preflight,
    _witness,
)


_MAX_ORDER = 65_536


class WitnessMinimizationDefinitionError(AnyAlgebraError, ValueError):
    """A witness-minimization request was malformed before evaluation."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid witness minimization {field}: {reason}")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise WitnessMinimizationDefinitionError(
            field=field, reason="must be a non-empty trimmed built-in str"
        )
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DeclaredWitnessOrder:
    """An inert bounded declaration; permutation validity needs a domain."""

    name: str
    assignment_indices: tuple[int, ...]
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        name: str,
        assignment_indices: Iterable[int],
        algorithm: str,
        algorithm_version: int,
    ) -> None:
        if type(self) is not DeclaredWitnessOrder:
            raise WitnessMinimizationDefinitionError(
                field="order", reason="must be an exact DeclaredWitnessOrder"
            )
        try:
            iterator = iter(assignment_indices)
        except Exception:
            raise WitnessMinimizationDefinitionError(
                field="assignment_indices", reason="could not be snapshotted"
            ) from None
        values_list: list[int] = []
        for _ in range(_MAX_ORDER + 1):
            try:
                values_list.append(next(iterator))
            except StopIteration:
                break
            except Exception:
                raise WitnessMinimizationDefinitionError(
                    field="assignment_indices", reason="could not be snapshotted"
                ) from None
        values = tuple(values_list)
        if len(values) > _MAX_ORDER:
            raise WitnessMinimizationDefinitionError(
                field="assignment_indices", reason="exceeds declared maximum 65536"
            )
        for value in values:
            if type(value) is not int or value < 0:
                raise WitnessMinimizationDefinitionError(
                    field="assignment_indices",
                    reason="must contain non-negative built-in ints",
                )
        if type(algorithm_version) is not int or algorithm_version <= 0:
            raise WitnessMinimizationDefinitionError(
                field="algorithm_version", reason="must be a positive built-in int"
            )
        object.__setattr__(self, "name", _text(name, "name"))
        object.__setattr__(self, "assignment_indices", values)
        object.__setattr__(self, "algorithm", _text(algorithm, "algorithm"))
        object.__setattr__(self, "algorithm_version", algorithm_version)

    @classmethod
    def lexicographic(cls, domain_size: int) -> DeclaredWitnessOrder:
        if cls is not DeclaredWitnessOrder:
            raise WitnessMinimizationDefinitionError(
                field="order", reason="factory requires exact DeclaredWitnessOrder"
            )
        if type(domain_size) is not int or domain_size <= 0 or domain_size > _MAX_ORDER:
            raise WitnessMinimizationDefinitionError(
                field="domain_size", reason="must be a positive bounded built-in int"
            )
        return cls("lexicographic", range(domain_size), "anyalgebra.lexicographic", 1)

    def __repr__(self) -> str:
        return (
            "DeclaredWitnessOrder("
            f"name={self.name!r}, size={len(self.assignment_indices)})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MinimizationRecord:
    structure: Structure
    law: Law
    source_report: Disproved | SampledDisproved
    source_witness: ValidationWitness
    source_assignment_index: int
    source_carrier_indices: tuple[int, ...]
    order: DeclaredWitnessOrder
    domain_size: int
    order_size: int
    candidates_evaluated: int
    prefix_indices: tuple[int, ...]
    evaluated_indices: tuple[int, ...]
    earlier_gap_index: int | None
    earlier_gap_witness: ValidationWitness | None
    reason: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise WitnessMinimizationDefinitionError(
            field="result", reason="is minimizer-owned"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(candidates_evaluated={self.candidates_evaluated}, "
            f"has_gap={self.earlier_gap_index is not None})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MinimizedWitness(MinimizationRecord):
    chosen_assignment_index: int
    chosen_carrier_indices: tuple[int, ...]
    witness: ValidationWitness


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MinimizationInconclusive(MinimizationRecord):
    pass


def _domain(structure: Structure, law: Law) -> FiniteSubstitutionDomain:
    try:
        return _preflight(structure, law)
    except ValidationDefinitionError as error:
        raise WitnessMinimizationDefinitionError(
            field=error.field, reason=error.reason
        ) from None


def _validate_order(
    order: DeclaredWitnessOrder, domain: FiniteSubstitutionDomain
) -> None:
    values = order.assignment_indices
    if len(values) != len(domain):
        raise WitnessMinimizationDefinitionError(
            field="assignment_indices",
            reason="must contain every domain assignment exactly once",
        )
    if set(values) != set(range(len(domain))):
        raise WitnessMinimizationDefinitionError(
            field="assignment_indices", reason="must be a full in-range permutation"
        )


def _snapshot_order(order: DeclaredWitnessOrder) -> DeclaredWitnessOrder:
    """Copy every validated declaration field before evaluation can mutate it."""
    try:
        name, indices, algorithm, version = (
            order.name,
            order.assignment_indices,
            order.algorithm,
            order.algorithm_version,
        )
    except AttributeError:
        raise WitnessMinimizationDefinitionError(
            field="order", reason="is missing required declaration metadata"
        ) from None
    if (
        type(name) is not str
        or not name
        or name != name.strip()
        or type(indices) is not tuple
        or len(indices) > _MAX_ORDER
        or any(type(index) is not int or index < 0 for index in indices)
        or type(algorithm) is not str
        or not algorithm
        or algorithm != algorithm.strip()
        or type(version) is not int
        or version <= 0
    ):
        raise WitnessMinimizationDefinitionError(
            field="order", reason="has malformed declaration metadata"
        )
    return DeclaredWitnessOrder(name, indices, algorithm, version)


def _result(
    kind: type[MinimizedWitness] | type[MinimizationInconclusive],
    structure: Structure,
    law: Law,
    source: Disproved | SampledDisproved,
    source_witness: ValidationWitness,
    order: DeclaredWitnessOrder,
    domain: FiniteSubstitutionDomain,
    source_assignment_index: int,
    source_carrier_indices: tuple[int, ...],
    evaluated: tuple[int, ...],
    *,
    reason: str,
    gap_index: int | None = None,
    gap: ValidationWitness | None = None,
    chosen_index: int | None = None,
    witness: ValidationWitness | None = None,
) -> MinimizedWitness | MinimizationInconclusive:
    value = object.__new__(kind)
    object.__setattr__(value, "structure", structure)
    object.__setattr__(value, "law", law)
    object.__setattr__(value, "source_report", source)
    object.__setattr__(value, "source_witness", source_witness)
    object.__setattr__(value, "source_assignment_index", source_assignment_index)
    object.__setattr__(value, "source_carrier_indices", source_carrier_indices)
    object.__setattr__(value, "order", order)
    object.__setattr__(value, "domain_size", len(domain))
    object.__setattr__(value, "order_size", len(order.assignment_indices))
    object.__setattr__(value, "candidates_evaluated", len(evaluated))
    object.__setattr__(value, "prefix_indices", evaluated)
    object.__setattr__(value, "evaluated_indices", evaluated)
    object.__setattr__(value, "earlier_gap_index", gap_index)
    object.__setattr__(value, "earlier_gap_witness", gap)
    object.__setattr__(value, "reason", reason)
    if kind is MinimizedWitness:
        assert chosen_index is not None and witness is not None
        object.__setattr__(value, "chosen_assignment_index", chosen_index)
        object.__setattr__(
            value, "chosen_carrier_indices", witness.substitution_indices
        )
        object.__setattr__(value, "witness", witness)
    return value


def _source_coordinates(
    source: Disproved | SampledDisproved, domain: FiniteSubstitutionDomain
) -> tuple[int, tuple[int, ...], ValidationWitness]:
    """Validate a sealed negative witness against the current finite domain."""
    try:
        raw_witness = source.witness
        if type(raw_witness) is not ValidationWitness:
            raise WitnessMinimizationDefinitionError(
                field="source", reason="has malformed definite conclusion witness"
            )
        witness = raw_witness
        indices = witness.substitution_indices
        phase, premise_index = witness.phase, witness.premise_index
        left, right = witness.left, witness.right
        report_law = source.law
        if type(source) is Disproved:
            exact = source
            (
                expected,
                evaluated,
                enumeration,
                carrier_sizes,
                algorithm,
                version,
                semantics,
            ) = (
                exact.expected_assignments,
                exact.evaluated_assignments,
                exact.enumeration_policy,
                exact.variable_carrier_sizes,
                exact.algorithm,
                exact.algorithm_version,
                exact.partial_semantics,
            )
            expected_carrier_sizes = tuple(
                len(carrier) for carrier in domain._carriers_by_variable()
            )
            report_ok = (
                type(report_law) is Law
                and type(expected) is int
                and expected == len(domain)
                and type(evaluated) is int
                and 0 < evaluated <= len(domain)
                and type(enumeration) is str
                and enumeration == "finite_substitution_domain_lexicographic"
                and type(carrier_sizes) is tuple
                and all(type(size) is int and size > 0 for size in carrier_sizes)
                and carrier_sizes == expected_carrier_sizes
                and type(algorithm) is str
                and algorithm == "anyalgebra.exhaustive_finite_law"
                and type(version) is int
                and version == 1
                and type(semantics) is str
                and semantics == report_law.partial_semantics
            )
        else:
            sampled = cast(SampledDisproved, source)
            (
                requested,
                evaluated_samples,
                planned,
                evaluated_indices,
                domain_size,
                seed,
                algorithm,
                version,
                semantics,
                reason,
            ) = (
                sampled.requested_samples,
                sampled.evaluated_samples,
                sampled.planned_indices,
                sampled.evaluated_indices,
                sampled.domain_size,
                sampled.seed,
                sampled.algorithm,
                sampled.algorithm_version,
                sampled.partial_semantics,
                sampled.reason,
            )
            report_ok = (
                type(report_law) is Law
                and type(requested) is int
                and 0 < requested <= _MAX_ORDER
                and requested <= len(domain)
                and type(evaluated_samples) is int
                and 0 < evaluated_samples <= requested
                and type(seed) is int
                and 0 <= seed < (1 << 64)
                and type(planned) is tuple
                and type(evaluated_indices) is tuple
                and type(domain_size) is int
                and domain_size == len(domain)
                and len(planned) == requested
                and len(evaluated_indices) == evaluated_samples
                and evaluated_indices == planned[: len(evaluated_indices)]
                and all(
                    type(item) is int and 0 <= item < len(domain) for item in planned
                )
                and len(set(planned)) == len(planned)
                and all(
                    type(item) is int and 0 <= item < len(domain)
                    for item in evaluated_indices
                )
                and len(set(evaluated_indices)) == len(evaluated_indices)
                and type(algorithm) is str
                and algorithm == "anyalgebra.splitmix64_partial_fisher_yates"
                and type(version) is int
                and version == 1
                and type(semantics) is str
                and semantics == report_law.partial_semantics
                and type(reason) is str
                and reason == "sampled_counterexample"
                and planned
                == _sample_indices(len(domain), SamplingOptions(requested, seed))
            )
    except AttributeError:
        raise WitnessMinimizationDefinitionError(
            field="source", reason="is missing required negative-report metadata"
        ) from None
    left_kind = type(left.outcome) if type(left) is EvaluationResult else None
    right_kind = type(right.outcome) if type(right) is EvaluationResult else None
    definite_outcomes = (left_kind is Defined and right_kind is Defined) or (
        semantics == "strong" and (left_kind is Undefined or right_kind is Undefined)
    )
    if (
        type(indices) is not tuple
        or type(phase) is not str
        or phase != "conclusion"
        or premise_index is not None
        or type(left) is not EvaluationResult
        or type(right) is not EvaluationResult
        or not report_ok
        or not definite_outcomes
    ):
        raise WitnessMinimizationDefinitionError(
            field="source", reason="has malformed definite conclusion witness"
        )
    carriers = domain._carriers_by_variable()
    if len(indices) != len(carriers):
        raise WitnessMinimizationDefinitionError(
            field="source", reason="witness carrier-index arity is invalid"
        )
    flat = 0
    for index, carrier in zip(indices, carriers, strict=True):
        if type(index) is not int or index < 0 or index >= len(carrier):
            raise WitnessMinimizationDefinitionError(
                field="source", reason="witness carrier indices are out of range"
            )
        flat = flat * len(carrier) + index
    if type(source) is Disproved:
        if flat != evaluated - 1:
            raise WitnessMinimizationDefinitionError(
                field="source", reason="exhaustive witness index does not match report"
            )
    elif flat != evaluated_indices[-1]:
        raise WitnessMinimizationDefinitionError(
            field="source", reason="sampled witness index does not match report"
        )
    return flat, indices, _witness(indices, phase, premise_index, left, right)


def minimize_counterexample(
    structure: Structure,
    law: Law,
    source: Disproved | SampledDisproved,
    order: DeclaredWitnessOrder,
) -> MinimizedWitness | MinimizationInconclusive:
    """Search the declared total order, refusing to claim minimality past a gap."""
    if type(structure) is not Structure:
        raise WitnessMinimizationDefinitionError(
            field="structure", reason="must be an exact Structure"
        )
    if type(law) is not Law:
        raise WitnessMinimizationDefinitionError(
            field="law", reason="must be an exact Law"
        )
    if type(order) is not DeclaredWitnessOrder:
        raise WitnessMinimizationDefinitionError(
            field="order", reason="must be an exact DeclaredWitnessOrder"
        )
    if type(source) not in (Disproved, SampledDisproved):
        raise WitnessMinimizationDefinitionError(
            field="source", reason="must be an exact negative validation report"
        )
    try:
        source_structure, source_law = (
            source.structure,
            source.law,
        )
    except AttributeError:
        raise WitnessMinimizationDefinitionError(
            field="source", reason="is missing required report metadata"
        ) from None
    if source_structure is not structure or source_law is not law:
        raise WitnessMinimizationDefinitionError(
            field="source", reason="has foreign or missing provenance witness"
        )
    domain = _domain(structure, law)
    source_assignment_index, source_carrier_indices, source_witness = (
        _source_coordinates(source, domain)
    )
    order_snapshot = _snapshot_order(order)
    _validate_order(order_snapshot, domain)
    evaluated: list[int] = []
    for index in order_snapshot.assignment_indices:
        evaluated.append(index)
        substitution = domain[index]
        gap: ValidationWitness | None = None
        candidate: ValidationWitness | None = None
        for premise_index, premise in enumerate(law.hypotheses):
            left, right = (
                _evaluate(structure, premise.left, substitution),
                _evaluate(structure, premise.right, substitution),
            )
            if type(left.outcome) is Undefined or type(right.outcome) is Undefined:
                break
            if type(left.outcome) is not Defined or type(right.outcome) is not Defined:
                gap = _witness(
                    substitution.carrier_indices, "premise", premise_index, left, right
                )
                break
            carrier = _carrier_for_sort(structure, premise.left.sort)
            assert carrier is not None
            equal = _equal_defined(carrier, left.outcome, right.outcome)
            if equal is None:
                gap = _witness(
                    substitution.carrier_indices,
                    "premise_comparison",
                    premise_index,
                    left,
                    right,
                )
                break
            if not equal:
                break
        else:
            left, right = (
                _evaluate(structure, law.conclusion.left, substitution),
                _evaluate(structure, law.conclusion.right, substitution),
            )
            if type(left.outcome) is Undefined or type(right.outcome) is Undefined:
                if law.partial_semantics == "strong":
                    candidate = _witness(
                        substitution.carrier_indices, "conclusion", None, left, right
                    )
            elif (
                type(left.outcome) is not Defined or type(right.outcome) is not Defined
            ):
                gap = _witness(
                    substitution.carrier_indices, "conclusion", None, left, right
                )
            else:
                carrier = _carrier_for_sort(structure, law.conclusion.left.sort)
                assert carrier is not None
                equal = _equal_defined(carrier, left.outcome, right.outcome)
                if equal is None:
                    gap = _witness(
                        substitution.carrier_indices,
                        "conclusion_comparison",
                        None,
                        left,
                        right,
                    )
                elif not equal:
                    candidate = _witness(
                        substitution.carrier_indices, "conclusion", None, left, right
                    )
        if gap is not None:
            return _result(
                MinimizationInconclusive,
                structure,
                law,
                source,
                source_witness,
                order_snapshot,
                domain,
                source_assignment_index,
                source_carrier_indices,
                tuple(evaluated),
                reason="earlier_evaluation_gap",
                gap_index=index,
                gap=gap,
            )
        if candidate is not None:
            return _result(
                MinimizedWitness,
                structure,
                law,
                source,
                source_witness,
                order_snapshot,
                domain,
                source_assignment_index,
                source_carrier_indices,
                tuple(evaluated),
                reason="first_definite_counterexample",
                chosen_index=index,
                witness=candidate,
            )
    return _result(
        MinimizationInconclusive,
        structure,
        law,
        source,
        source_witness,
        order_snapshot,
        domain,
        source_assignment_index,
        source_carrier_indices,
        tuple(evaluated),
        reason="no_current_counterexample",
    )
