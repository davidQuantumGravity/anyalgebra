"""Deterministic exhaustive filters for allow-listed census constraints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError

from .constraints import CensusConstraint, CensusTerm
from .reference import ReferenceCandidate
from .spec import CensusSpec, CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table


class ConstraintFilterError(AnyAlgebraError, ValueError):
    """Constraint compilation or candidate validation failed safely."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        required_work: int | None = None,
    ) -> None:
        self.field = field
        self.reason = reason
        self.required_work = required_work
        super().__init__(f"invalid census filter {field}: {reason}")


def _parameter(constraint: CensusConstraint, name: str) -> int | str:
    for key, value in constraint.parameters:
        if key == name:
            return value
    raise ConstraintFilterError(field="constraint", reason="missing parameter")


def _term_size(term: CensusTerm) -> int:
    return 1 + sum(_term_size(argument) for argument in term.arguments)


def _assignment_count(carrier_size: int, variable_count: int) -> int:
    result = 1
    for _ in range(variable_count):
        result *= carrier_size
    return result


def _constraint_work(constraint: CensusConstraint, core: CensusSpecCore) -> int:
    size = core.carrier_size
    if constraint.kind == "nullary_value":
        return 1
    if constraint.kind == "identity":
        sides = 2 if _parameter(constraint, "side") == "two_sided" else 1
        return sides * size
    if constraint.kind == "commutative":
        return size * size
    if constraint.kind == "idempotent":
        return size
    if constraint.kind == "quasigroup":
        return 2 * size * size
    if constraint.kind == "equation":
        assert constraint.left is not None and constraint.right is not None
        count = cast(int, _parameter(constraint, "variable_count"))
        return _assignment_count(size, count) * (
            _term_size(constraint.left) + _term_size(constraint.right)
        )
    raise ConstraintFilterError(field="constraint", reason="unknown constraint kind")


def _operation(
    outputs: tuple[int, ...], core: CensusSpecCore, arguments: tuple[int, ...]
) -> int:
    index = 0
    for argument in arguments:
        index = index * core.carrier_size + argument
    return outputs[index]


def _term(
    term: CensusTerm,
    assignment: tuple[int, ...],
    outputs: tuple[int, ...],
    core: CensusSpecCore,
) -> int:
    if term.kind == "variable":
        assert term.index is not None
        return assignment[term.index]
    if term.kind == "constant":
        assert term.index is not None
        return term.index
    return _operation(
        outputs,
        core,
        tuple(
            _term(argument, assignment, outputs, core) for argument in term.arguments
        ),
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConstraintCheck:
    """One exhaustive deterministic constraint result for one candidate."""

    constraint: CensusConstraint
    accepted: bool
    checked_cases: int
    violation_count: int
    first_witness: tuple[int, ...] | None
    reason: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintFilterError(field="check", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ConstraintCheck cannot be subclassed")

    @classmethod
    def _create(
        cls,
        constraint: CensusConstraint,
        *,
        checked_cases: int,
        violation_count: int,
        first_witness: tuple[int, ...] | None,
        failure_reason: str,
    ) -> ConstraintCheck:
        value = object.__new__(ConstraintCheck)
        for field, item in (
            ("constraint", constraint),
            ("accepted", violation_count == 0),
            ("checked_cases", checked_cases),
            ("violation_count", violation_count),
            ("first_witness", first_witness),
            ("reason", "satisfied" if violation_count == 0 else failure_reason),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is ConstraintCheck and (
            self.constraint,
            self.accepted,
            self.checked_cases,
            self.violation_count,
            self.first_witness,
            self.reason,
        ) == (
            other.constraint,
            other.accepted,
            other.checked_cases,
            other.violation_count,
            other.first_witness,
            other.reason,
        )

    def __repr__(self) -> str:
        return (
            "ConstraintCheck("
            f"kind={self.constraint.kind!r}, accepted={self.accepted}, "
            f"checked_cases={self.checked_cases}, "
            f"violation_count={self.violation_count})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CandidateFilterResult:
    """Complete conjunction of every canonical check for one raw candidate."""

    candidate: ReferenceCandidate
    checks: tuple[ConstraintCheck, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintFilterError(field="result", reason="is factory-owned")

    @classmethod
    def _create(
        cls, candidate: ReferenceCandidate, checks: tuple[ConstraintCheck, ...]
    ) -> CandidateFilterResult:
        reasons = tuple(check.reason for check in checks if not check.accepted)
        value = object.__new__(CandidateFilterResult)
        object.__setattr__(value, "candidate", candidate)
        object.__setattr__(value, "checks", checks)
        object.__setattr__(value, "accepted", not reasons)
        object.__setattr__(value, "rejection_reasons", reasons)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CandidateFilterResult and (
            self.candidate,
            self.checks,
            self.accepted,
            self.rejection_reasons,
        ) == (
            other.candidate,
            other.checks,
            other.accepted,
            other.rejection_reasons,
        )

    def __repr__(self) -> str:
        return (
            "CandidateFilterResult("
            f"candidate_index={self.candidate.candidate_index}, "
            f"accepted={self.accepted}, check_count={len(self.checks)})"
        )


def _check(
    constraint: CensusConstraint,
    outputs: tuple[int, ...],
    core: CensusSpecCore,
) -> ConstraintCheck:
    size = core.carrier_size
    violation_count = 0
    first_witness: tuple[int, ...] | None = None

    def record_violation(witness: tuple[int, ...]) -> None:
        nonlocal violation_count, first_witness
        violation_count += 1
        if first_witness is None:
            first_witness = witness

    if constraint.kind == "nullary_value":
        element = cast(int, _parameter(constraint, "element"))
        if outputs[0] != element:
            record_violation(())
        return ConstraintCheck._create(
            constraint,
            checked_cases=1,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="nullary_value_mismatch",
        )
    if constraint.kind == "identity":
        element = cast(int, _parameter(constraint, "element"))
        side = cast(str, _parameter(constraint, "side"))
        sides = (
            ((0, True), (1, False))
            if side == "two_sided"
            else ((0, True),)
            if side == "left"
            else ((1, False),)
        )
        for side_code, left in sides:
            for item in range(size):
                arguments = (element, item) if left else (item, element)
                if _operation(outputs, core, arguments) != item:
                    record_violation((side_code, item))
        return ConstraintCheck._create(
            constraint,
            checked_cases=len(sides) * size,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="identity_mismatch",
        )
    if constraint.kind == "commutative":
        for left_element, right_element in product(range(size), repeat=2):
            if _operation(outputs, core, (left_element, right_element)) != _operation(
                outputs, core, (right_element, left_element)
            ):
                record_violation((left_element, right_element))
        return ConstraintCheck._create(
            constraint,
            checked_cases=size * size,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="commutativity_mismatch",
        )
    if constraint.kind == "idempotent":
        for item in range(size):
            if _operation(outputs, core, (item, item)) != item:
                record_violation((item,))
        return ConstraintCheck._create(
            constraint,
            checked_cases=size,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="idempotence_mismatch",
        )
    if constraint.kind == "quasigroup":
        carrier = set(range(size))
        for axis in range(2):
            for fixed in range(size):
                image = {
                    _operation(
                        outputs,
                        core,
                        (fixed, moving) if axis == 0 else (moving, fixed),
                    )
                    for moving in range(size)
                }
                if image != carrier:
                    record_violation((axis, fixed))
        return ConstraintCheck._create(
            constraint,
            checked_cases=2 * size * size,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="quasigroup_translation_mismatch",
        )
    if constraint.kind == "equation":
        count = cast(int, _parameter(constraint, "variable_count"))
        assert constraint.left is not None and constraint.right is not None
        assignments = product(range(size), repeat=count)
        checked = _assignment_count(size, count)
        for assignment in assignments:
            if _term(constraint.left, assignment, outputs, core) != _term(
                constraint.right, assignment, outputs, core
            ):
                record_violation(assignment)
        return ConstraintCheck._create(
            constraint,
            checked_cases=checked,
            violation_count=violation_count,
            first_witness=first_witness,
            failure_reason="equation_mismatch",
        )
    raise ConstraintFilterError(field="constraint", reason="unknown constraint kind")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CompiledConstraintFilter:
    """Core-owned immutable filter with a preflighted exact work charge."""

    spec: CensusSpec
    estimated_work_per_candidate: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintFilterError(field="filter", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CompiledConstraintFilter cannot be subclassed")

    @classmethod
    def _create(cls, spec: CensusSpec, work: int) -> CompiledConstraintFilter:
        value = object.__new__(CompiledConstraintFilter)
        object.__setattr__(value, "spec", spec)
        object.__setattr__(value, "estimated_work_per_candidate", work)
        return value

    def evaluate(self, candidate: ReferenceCandidate) -> CandidateFilterResult:
        if type(candidate) is not ReferenceCandidate:
            raise ConstraintFilterError(
                field="candidate", reason="must be exact ReferenceCandidate"
            )
        try:
            rank = rank_operation_table(self.spec.core, candidate.outputs)
        except OperationTableCodeError as error:
            raise ConstraintFilterError(
                field="candidate", reason="does not belong to the filter core"
            ) from error
        if rank != candidate.candidate_index:
            raise ConstraintFilterError(
                field="candidate", reason="index does not match table outputs"
            )
        checks = tuple(
            _check(constraint, candidate.outputs, self.spec.core)
            for constraint in self.spec.constraints.constraints
        )
        return CandidateFilterResult._create(candidate, checks)


def compile_constraint_filter(spec: CensusSpec) -> CompiledConstraintFilter:
    """Compile and preflight all canonical constraints without evaluating a table."""
    if type(spec) is not CensusSpec:
        raise ConstraintFilterError(field="spec", reason="must be exact CensusSpec")
    work = sum(
        _constraint_work(constraint, spec.core)
        for constraint in spec.constraints.constraints
    )
    if work > spec.bounds.max_work_units:
        raise ConstraintFilterError(
            field="max_work_units",
            reason="constraint checks exceed the declared per-candidate bound",
            required_work=work,
        )
    return CompiledConstraintFilter._create(spec, work)


__all__ = (
    "CandidateFilterResult",
    "CompiledConstraintFilter",
    "ConstraintCheck",
    "ConstraintFilterError",
    "compile_constraint_filter",
)
