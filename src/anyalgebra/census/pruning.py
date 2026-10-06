"""Sound local-prefix pruning for the bounded reference census."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError

from .constraints import CensusConstraint, CensusTerm
from .filtering import CompiledConstraintFilter, compile_constraint_filter
from .reference import REFERENCE_BATCH_LIMIT, ReferenceCandidate
from .spec import CensusSpec, CensusSpecCore
from .table_codes import rank_operation_table


class PrefixPruningError(AnyAlgebraError, ValueError):
    """A prefix, pruning preflight, or pruning record was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid prefix pruning {field}: {reason}")


def _parameter(constraint: CensusConstraint, name: str) -> int | str:
    for key, value in constraint.parameters:
        if key == name:
            return value
    raise PrefixPruningError(field="constraint", reason="missing parameter")


def _prefix(spec: CensusSpec, value: object) -> tuple[int, ...]:
    if isinstance(value, str | bytes):
        raise PrefixPruningError(
            field="prefix", reason="must be an iterable of exact table digits"
        )
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise PrefixPruningError(
            field="prefix", reason="must be an iterable of exact table digits"
        ) from error
    result: list[int] = []
    try:
        for index in range(spec.core.input_tuple_count + 1):
            try:
                digit = next(iterator)
            except StopIteration:
                return tuple(result)
            if index == spec.core.input_tuple_count:
                raise PrefixPruningError(
                    field="prefix", reason="is longer than the complete table"
                )
            if type(digit) is not int:
                raise PrefixPruningError(
                    field="prefix",
                    reason="digits must be exact built-in ints, excluding bool",
                )
            if digit < 0 or digit >= spec.carrier_size:
                raise PrefixPruningError(
                    field="prefix", reason="digit is outside the carrier range"
                )
            result.append(digit)
    except PrefixPruningError:
        raise
    except Exception as error:
        raise PrefixPruningError(
            field="prefix", reason="iterable traversal failed"
        ) from error
    raise AssertionError("bounded prefix snapshot did not terminate")


def _partial_operation(
    prefix: tuple[int, ...], core: CensusSpecCore, arguments: tuple[int, ...]
) -> int | None:
    index = 0
    for argument in arguments:
        index = index * core.carrier_size + argument
    if index >= len(prefix):
        return None
    return prefix[index]


def _partial_term(
    term: CensusTerm,
    assignment: tuple[int, ...],
    prefix: tuple[int, ...],
    core: CensusSpecCore,
) -> int | None:
    if term.kind == "variable":
        assert term.index is not None
        return assignment[term.index]
    if term.kind == "constant":
        assert term.index is not None
        return term.index
    arguments: list[int] = []
    for child in term.arguments:
        value = _partial_term(child, assignment, prefix, core)
        if value is None:
            return None
        arguments.append(value)
    return _partial_operation(prefix, core, tuple(arguments))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PrefixViolation:
    """One canonical already-certain constraint violation in a prefix."""

    constraint: CensusConstraint
    reason: str
    witness: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise PrefixPruningError(field="violation", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("PrefixViolation cannot be subclassed")

    @classmethod
    def _create(
        cls, constraint: CensusConstraint, reason: str, witness: tuple[int, ...]
    ) -> PrefixViolation:
        value = object.__new__(PrefixViolation)
        object.__setattr__(value, "constraint", constraint)
        object.__setattr__(value, "reason", reason)
        object.__setattr__(value, "witness", witness)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is PrefixViolation and (
            self.constraint,
            self.reason,
            self.witness,
        ) == (other.constraint, other.reason, other.witness)


def _violation(
    constraint: CensusConstraint,
    prefix: tuple[int, ...],
    core: CensusSpecCore,
) -> PrefixViolation | None:
    size = core.carrier_size
    if constraint.kind == "nullary_value":
        expected = cast(int, _parameter(constraint, "element"))
        if prefix and prefix[0] != expected:
            return PrefixViolation._create(constraint, "nullary_value_mismatch", ())
        return None
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
        for side_code, is_left in sides:
            for item in range(size):
                arguments = (element, item) if is_left else (item, element)
                output = _partial_operation(prefix, core, arguments)
                if output is not None and output != item:
                    return PrefixViolation._create(
                        constraint, "identity_mismatch", (side_code, item)
                    )
        return None
    if constraint.kind == "commutative":
        for left_element, right_element in product(range(size), repeat=2):
            forward = _partial_operation(prefix, core, (left_element, right_element))
            reverse = _partial_operation(prefix, core, (right_element, left_element))
            if forward is not None and reverse is not None and forward != reverse:
                return PrefixViolation._create(
                    constraint,
                    "commutativity_mismatch",
                    (left_element, right_element),
                )
        return None
    if constraint.kind == "idempotent":
        for item in range(size):
            output = _partial_operation(prefix, core, (item, item))
            if output is not None and output != item:
                return PrefixViolation._create(
                    constraint, "idempotence_mismatch", (item,)
                )
        return None
    if constraint.kind == "quasigroup":
        for axis in range(2):
            for fixed in range(size):
                first_by_output: dict[int, int] = {}
                for moving in range(size):
                    arguments = (fixed, moving) if axis == 0 else (moving, fixed)
                    output = _partial_operation(prefix, core, arguments)
                    if output is None:
                        continue
                    prior = first_by_output.get(output)
                    if prior is not None:
                        return PrefixViolation._create(
                            constraint,
                            "quasigroup_duplicate",
                            (axis, fixed, prior, moving, output),
                        )
                    first_by_output[output] = moving
        return None
    if constraint.kind == "equation":
        count = cast(int, _parameter(constraint, "variable_count"))
        assert constraint.left is not None and constraint.right is not None
        for assignment in product(range(size), repeat=count):
            left_value = _partial_term(constraint.left, assignment, prefix, core)
            right_value = _partial_term(constraint.right, assignment, prefix, core)
            if (
                left_value is not None
                and right_value is not None
                and left_value != right_value
            ):
                return PrefixViolation._create(
                    constraint, "equation_mismatch", assignment
                )
        return None
    raise PrefixPruningError(field="constraint", reason="unknown constraint kind")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PrefixPruningDecision:
    """Replayable decision for one exact lexicographic table prefix."""

    prefix: tuple[int, ...]
    violations: tuple[PrefixViolation, ...]
    pruned: bool
    skipped_candidate_count: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise PrefixPruningError(field="decision", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("PrefixPruningDecision cannot be subclassed")

    @classmethod
    def _create(
        cls,
        prefix: tuple[int, ...],
        violations: tuple[PrefixViolation, ...],
        core: CensusSpecCore,
    ) -> PrefixPruningDecision:
        remaining = core.input_tuple_count - len(prefix)
        skipped = core.carrier_size**remaining if violations else 0
        value = object.__new__(PrefixPruningDecision)
        object.__setattr__(value, "prefix", prefix)
        object.__setattr__(value, "violations", violations)
        object.__setattr__(value, "pruned", bool(violations))
        object.__setattr__(value, "skipped_candidate_count", skipped)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is PrefixPruningDecision and (
            self.prefix,
            self.violations,
            self.pruned,
            self.skipped_candidate_count,
        ) == (
            other.prefix,
            other.violations,
            other.pruned,
            other.skipped_candidate_count,
        )

    def __repr__(self) -> str:
        return (
            "PrefixPruningDecision("
            f"prefix_length={len(self.prefix)}, pruned={self.pruned}, "
            f"violation_count={len(self.violations)}, "
            f"skipped_candidate_count_bits={self.skipped_candidate_count.bit_length()})"
        )


def _decision(spec: CensusSpec, prefix: tuple[int, ...]) -> PrefixPruningDecision:
    violations = tuple(
        violation
        for constraint in spec.constraints.constraints
        if (violation := _violation(constraint, prefix, spec.core)) is not None
    )
    return PrefixPruningDecision._create(prefix, violations, spec.core)


def check_prefix(spec: CensusSpec, prefix: Iterable[int]) -> PrefixPruningDecision:
    """Return every currently certain local violation for one bounded prefix."""
    if type(spec) is not CensusSpec:
        raise PrefixPruningError(field="spec", reason="must be exact CensusSpec")
    compile_constraint_filter(spec)
    return _decision(spec, _prefix(spec, prefix))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PrunedEnumeration:
    """Complete small-corpus DFS result with replayable pruned prefixes."""

    spec: CensusSpec
    accepted_candidates: tuple[ReferenceCandidate, ...]
    events: tuple[PrefixPruningDecision, ...]
    examined_candidate_count: int
    skipped_candidate_count: int
    complete: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise PrefixPruningError(field="enumeration", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("PrunedEnumeration cannot be subclassed")

    @classmethod
    def _create(
        cls,
        spec: CensusSpec,
        accepted: tuple[ReferenceCandidate, ...],
        events: tuple[PrefixPruningDecision, ...],
        examined: int,
    ) -> PrunedEnumeration:
        skipped = sum(event.skipped_candidate_count for event in events)
        value = object.__new__(PrunedEnumeration)
        object.__setattr__(value, "spec", spec)
        object.__setattr__(value, "accepted_candidates", accepted)
        object.__setattr__(value, "events", events)
        object.__setattr__(value, "examined_candidate_count", examined)
        object.__setattr__(value, "skipped_candidate_count", skipped)
        object.__setattr__(
            value, "complete", examined + skipped == spec.core.candidate_count
        )
        return value

    def __repr__(self) -> str:
        return (
            "PrunedEnumeration("
            f"accepted_count={len(self.accepted_candidates)}, "
            f"event_count={len(self.events)}, "
            f"examined_candidate_count={self.examined_candidate_count}, "
            f"skipped_candidate_count={self.skipped_candidate_count}, "
            f"complete={self.complete})"
        )


def _enumeration_preflight(
    spec: CensusSpec, compiled: CompiledConstraintFilter
) -> None:
    total = spec.core.candidate_count
    if total > spec.bounds.max_candidates:
        raise PrefixPruningError(
            field="max_candidates", reason="complete differential corpus exceeds bound"
        )
    if total > REFERENCE_BATCH_LIMIT:
        raise PrefixPruningError(
            field="implementation_batch_limit",
            reason="complete differential corpus exceeds safety limit",
        )
    record_charge = 128 + 16 * spec.core.input_tuple_count
    if total * record_charge > spec.bounds.max_memory_bytes:
        raise PrefixPruningError(
            field="max_memory_bytes",
            reason="worst-case retained differential result exceeds bound",
        )
    required = total * max(compiled.estimated_work_per_candidate, 1)
    if required > spec.bounds.max_work_units:
        raise PrefixPruningError(
            field="max_work_units", reason="complete differential work exceeds bound"
        )


def enumerate_pruned(spec: CensusSpec) -> PrunedEnumeration:
    """Depth-first enumerate a complete bounded corpus with sound local pruning."""
    if type(spec) is not CensusSpec:
        raise PrefixPruningError(field="spec", reason="must be exact CensusSpec")
    compiled = compile_constraint_filter(spec)
    _enumeration_preflight(spec, compiled)
    if spec.core.candidate_count == 0:
        return PrunedEnumeration._create(spec, (), (), 0)

    accepted: list[ReferenceCandidate] = []
    events: list[PrefixPruningDecision] = []
    examined = 0

    def visit(prefix: tuple[int, ...]) -> None:
        nonlocal examined
        decision = _decision(spec, prefix)
        if decision.pruned:
            events.append(decision)
            return
        if len(prefix) == spec.core.input_tuple_count:
            examined += 1
            index = rank_operation_table(spec.core, prefix)
            candidate = ReferenceCandidate._create(spec.core, index)
            if compiled.evaluate(candidate).accepted:
                accepted.append(candidate)
            return
        for digit in range(spec.carrier_size):
            visit((*prefix, digit))

    visit(())
    return PrunedEnumeration._create(spec, tuple(accepted), tuple(events), examined)


__all__ = (
    "PrefixPruningDecision",
    "PrefixPruningError",
    "PrefixViolation",
    "PrunedEnumeration",
    "check_prefix",
    "enumerate_pruned",
)
