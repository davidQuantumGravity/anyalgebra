"""Deterministic bounded sampling of finite many-sorted equation laws.

Sampling is deliberately a weaker route than exhaustive validation: even a
sample that covers every currently finite assignment records only sampled
evidence and never returns :class:`Proved`.
"""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.structures.laws import Law
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.structure import Structure
from anyalgebra.validation.domains import FiniteSubstitutionDomain
from anyalgebra.validation.validate import (
    ValidationDefinitionError,
    ValidationWitness,
    _carrier_for_sort,
    _equal_defined,
    _evaluate,
    _preflight,
    _witness,
)


_MAX_SAMPLES = 65_536
_UINT64_LIMIT = 1 << 64
_ALGORITHM = "anyalgebra.splitmix64_partial_fisher_yates"
_ALGORITHM_VERSION = 1


class SamplingDefinitionError(AnyAlgebraError, ValueError):
    """A sampled validation declaration failed before user operations ran."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid sampled validation {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SamplingOptions:
    """Exact bounded draw count and package-owned PRNG seed."""

    sample_count: int
    seed: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, sample_count: int, seed: int) -> None:
        if type(self) is not SamplingOptions:
            raise SamplingDefinitionError(
                field="options", reason="must be an exact SamplingOptions"
            )
        if type(sample_count) is not int or sample_count <= 0:
            raise SamplingDefinitionError(
                field="sample_count", reason="must be a positive built-in int"
            )
        if sample_count > _MAX_SAMPLES:
            raise SamplingDefinitionError(
                field="sample_count", reason="exceeds declared maximum 65536"
            )
        if type(seed) is not int or seed < 0 or seed >= _UINT64_LIMIT:
            raise SamplingDefinitionError(
                field="seed", reason="must be a built-in uint64"
            )
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "seed", seed)

    def __repr__(self) -> str:
        return f"SamplingOptions(sample_count={self.sample_count}, seed={self.seed})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SamplingReport:
    """Sealed replay metadata shared by negative and inconclusive samples."""

    structure: Structure
    law: Law
    domain_size: int
    requested_samples: int
    evaluated_samples: int
    planned_indices: tuple[int, ...]
    evaluated_indices: tuple[int, ...]
    seed: int
    algorithm: str
    algorithm_version: int
    premise_evaluations: int
    conclusion_evaluations: int
    vacuous_assignments: int
    skipped_assignments: int
    undecidable_assignments: int
    partial_semantics: str
    reason: str
    witness: ValidationWitness | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SamplingDefinitionError(field="report", reason="is sampler-owned")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(domain_size={self.domain_size}, "
            f"requested_samples={self.requested_samples}, "
            f"evaluated_samples={self.evaluated_samples}, "
            f"has_witness={self.witness is not None})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SampledDisproved(SamplingReport):
    """One sampled assignment supplied a definite replayable counterexample."""


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SampledInconclusive(SamplingReport):
    """No sampled counterexample was found; sampling is not a proof route."""


def _next_uint64(state: int) -> tuple[int, int]:
    """Advance fixed SplitMix64 state without Python's runtime PRNG behavior."""
    state = (state + 0x9E3779B97F4A7C15) & (_UINT64_LIMIT - 1)
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & (_UINT64_LIMIT - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & (_UINT64_LIMIT - 1)
    return state, value ^ (value >> 31)


def _sample_indices(domain_size: int, options: SamplingOptions) -> tuple[int, ...]:
    """Draw unique partial-Fisher--Yates indices in deterministic draw order."""
    state = options.seed
    swaps: dict[int, int] = {}
    chosen: list[int] = []
    for draw in range(options.sample_count):
        remaining = domain_size - draw
        limit = _UINT64_LIMIT - (_UINT64_LIMIT % remaining)
        while True:
            state, random_value = _next_uint64(state)
            if random_value < limit:
                break
        position = random_value % remaining
        selected = swaps.get(position, position)
        final_position = remaining - 1
        swaps[position] = swaps.get(final_position, final_position)
        chosen.append(selected)
    return tuple(chosen)


def _checked_domain(
    structure: Structure, law: Law, options: SamplingOptions
) -> FiniteSubstitutionDomain:
    """Reuse exhaustive preflight and translate only its stable declaration error."""
    try:
        domain = _preflight(structure, law)
    except ValidationDefinitionError as error:
        raise SamplingDefinitionError(field=error.field, reason=error.reason) from None
    if options.sample_count > len(domain):
        raise SamplingDefinitionError(
            field="sample_count", reason="must not exceed the finite domain size"
        )
    return domain


def _report(
    kind: type[SampledDisproved] | type[SampledInconclusive],
    structure: Structure,
    law: Law,
    options: SamplingOptions,
    domain_size: int,
    planned_indices: tuple[int, ...],
    *,
    evaluated: int,
    premises: int,
    conclusions: int,
    vacuous: int,
    skipped: int,
    undecidable: int,
    reason: str,
    witness: ValidationWitness | None,
) -> SampledDisproved | SampledInconclusive:
    """Allocate one sealed result after all sampler-owned metadata is known."""
    value = object.__new__(kind)
    object.__setattr__(value, "structure", structure)
    object.__setattr__(value, "law", law)
    object.__setattr__(value, "domain_size", domain_size)
    object.__setattr__(value, "requested_samples", options.sample_count)
    object.__setattr__(value, "evaluated_samples", evaluated)
    object.__setattr__(value, "planned_indices", planned_indices)
    object.__setattr__(value, "evaluated_indices", planned_indices[:evaluated])
    object.__setattr__(value, "seed", options.seed)
    object.__setattr__(value, "algorithm", _ALGORITHM)
    object.__setattr__(value, "algorithm_version", _ALGORITHM_VERSION)
    object.__setattr__(value, "premise_evaluations", premises)
    object.__setattr__(value, "conclusion_evaluations", conclusions)
    object.__setattr__(value, "vacuous_assignments", vacuous)
    object.__setattr__(value, "skipped_assignments", skipped)
    object.__setattr__(value, "undecidable_assignments", undecidable)
    object.__setattr__(value, "partial_semantics", law.partial_semantics)
    object.__setattr__(value, "reason", reason)
    object.__setattr__(value, "witness", witness)
    return value


def sample_law(
    structure: Structure, law: Law, options: SamplingOptions
) -> SampledDisproved | SampledInconclusive:
    """Sample a finite law without ever turning a sampled pass into a proof."""
    if type(structure) is not Structure:
        raise SamplingDefinitionError(
            field="structure", reason="must be an exact Structure"
        )
    if type(law) is not Law:
        raise SamplingDefinitionError(field="law", reason="must be an exact Law")
    if type(options) is not SamplingOptions:
        raise SamplingDefinitionError(
            field="options", reason="must be an exact SamplingOptions"
        )
    domain = _checked_domain(structure, law, options)
    planned_indices = _sample_indices(len(domain), options)
    evaluated = premises = conclusions = vacuous = skipped = undecidable = 0
    first_inconclusive: ValidationWitness | None = None
    for sampled_index in planned_indices:
        substitution = domain[sampled_index]
        evaluated += 1
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
                        substitution.carrier_indices,
                        "premise",
                        premise_index,
                        left,
                        right,
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
                        substitution.carrier_indices,
                        "premise_comparison",
                        premise_index,
                        left,
                        right,
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
                    SampledDisproved,
                    structure,
                    law,
                    options,
                    len(domain),
                    planned_indices,
                    evaluated=evaluated,
                    premises=premises,
                    conclusions=conclusions,
                    vacuous=vacuous,
                    skipped=skipped,
                    undecidable=undecidable,
                    reason="sampled_counterexample",
                    witness=_witness(
                        substitution.carrier_indices, "conclusion", None, left, right
                    ),
                )
            skipped += 1
            continue
        if type(left.outcome) is not Defined or type(right.outcome) is not Defined:
            undecidable += 1
            if first_inconclusive is None:
                first_inconclusive = _witness(
                    substitution.carrier_indices, "conclusion", None, left, right
                )
            continue
        carrier = _carrier_for_sort(structure, law.conclusion.left.sort)
        assert carrier is not None
        equal = _equal_defined(carrier, left.outcome, right.outcome)
        if equal is None:
            undecidable += 1
            if first_inconclusive is None:
                first_inconclusive = _witness(
                    substitution.carrier_indices,
                    "conclusion_comparison",
                    None,
                    left,
                    right,
                )
            continue
        if not equal:
            return _report(
                SampledDisproved,
                structure,
                law,
                options,
                len(domain),
                planned_indices,
                evaluated=evaluated,
                premises=premises,
                conclusions=conclusions,
                vacuous=vacuous,
                skipped=skipped,
                undecidable=undecidable,
                reason="sampled_counterexample",
                witness=_witness(
                    substitution.carrier_indices, "conclusion", None, left, right
                ),
            )
    return _report(
        SampledInconclusive,
        structure,
        law,
        options,
        len(domain),
        planned_indices,
        evaluated=evaluated,
        premises=premises,
        conclusions=conclusions,
        vacuous=vacuous,
        skipped=skipped,
        undecidable=undecidable,
        reason=(
            "sampled_evaluation_inconclusive"
            if first_inconclusive is not None
            else "sampled_pass_not_proof"
        ),
        witness=first_inconclusive,
    )
