"""Contract tests for deterministic non-proof sampled law validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.evaluate import EvaluationResult
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.sampling import (
    SampledDisproved,
    SampledInconclusive,
    SamplingDefinitionError,
    SamplingOptions,
    SamplingReport,
    _sample_indices,
    sample_law,
)
import anyalgebra.validation.sampling as sampling_module


def _unary_structure(
    function: Callable[[object], object], *, partial: bool = False
) -> tuple[Structure, Sort, OperationSymbol, Variable, list[int]]:
    sort = Sort("scalar")
    carrier = FiniteCarrier((0, 1), sort=sort)
    unary = OperationSymbol("unary", (sort,), sort)
    calls = [0]

    def counted(value: object) -> object:
        calls[0] += 1
        return function(value)

    operation: Operation | PartialOperation
    if partial:
        operation = PartialOperation.from_callable(unary, ((sort, carrier),), counted)
    else:
        operation = Operation.from_callable(unary, ((sort, carrier),), counted)
    structure = (
        StructureBuilder(Signature((sort,), (unary,)))
        .with_carrier(sort, carrier)
        .with_operation(unary, operation)
        .freeze()
    )
    return structure, sort, unary, Variable("x", sort), calls


def _unary_law(
    symbol: OperationSymbol, variable: Variable, *, partial: str = "strong"
) -> Law:
    term = Term.variable(variable)
    return Law(
        "sampled_unary",
        (variable,),
        Equation(Term.apply(symbol, term), term),
        partial_semantics=partial,
    )


def test_pinned_prng_schedule_and_report_determinism_are_stable() -> None:
    assert _sample_indices(5, SamplingOptions(5, 0)) == (0, 4, 1, 3, 2)
    structure, _sort, unary, variable, calls = _unary_structure(lambda value: value)
    law = _unary_law(unary, variable)
    first = sample_law(structure, law, SamplingOptions(2, 42))
    second = sample_law(structure, law, SamplingOptions(2, 42))

    assert type(first) is type(second) is SampledInconclusive
    assert first.planned_indices == second.planned_indices
    assert first.evaluated_indices == second.evaluated_indices == first.planned_indices
    assert first.reason == second.reason == "sampled_pass_not_proof"
    assert repr(first) == repr(second)
    assert first.algorithm == "anyalgebra.splitmix64_partial_fisher_yates"
    assert first.algorithm_version == 1
    assert calls == [4]


def test_different_seeds_unique_draws_and_full_domain_remain_inconclusive() -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(lambda value: value)
    law = _unary_law(unary, variable)
    first = sample_law(structure, law, SamplingOptions(2, 0))
    second = sample_law(structure, law, SamplingOptions(2, 2))

    assert type(first) is type(second) is SampledInconclusive
    assert first.planned_indices != second.planned_indices
    assert len(set(first.planned_indices)) == len(first.planned_indices) == 2
    assert first.domain_size == first.requested_samples == first.evaluated_samples == 2


def test_known_counterexample_can_be_found_or_missed_by_the_seed() -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(lambda _value: 0)
    law = _unary_law(unary, variable)
    missed = sample_law(structure, law, SamplingOptions(1, 2))
    found = sample_law(structure, law, SamplingOptions(1, 0))

    assert type(missed) is SampledInconclusive
    assert missed.planned_indices == missed.evaluated_indices == (0,)
    assert missed.reason == "sampled_pass_not_proof"
    assert type(found) is SampledDisproved
    assert found.planned_indices == found.evaluated_indices == (1,)
    assert found.witness is not None and found.witness.substitution_indices == (1,)


def test_early_counterexample_retains_only_the_evaluated_prefix() -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(lambda _value: 0)
    report = sample_law(structure, _unary_law(unary, variable), SamplingOptions(2, 0))

    assert type(report) is SampledDisproved
    assert report.planned_indices == (1, 0)
    assert report.evaluated_indices == (1,)
    assert report.evaluated_samples == len(report.evaluated_indices) == 1


def test_indeterminate_or_failed_sample_does_not_hide_a_later_disproof() -> None:
    def outcome(value: object) -> object:
        if value == 1:
            return Indeterminate("bounded")
        return 1

    structure, _sort, unary, variable, _calls = _unary_structure(outcome)
    report = sample_law(structure, _unary_law(unary, variable), SamplingOptions(2, 0))

    assert type(report) is SampledDisproved
    assert report.evaluated_indices == (1, 0)
    assert report.undecidable_assignments == 1
    assert report.witness is not None and report.witness.substitution_indices == (0,)

    failed_structure, _sort, failed_unary, failed_variable, _calls = _unary_structure(
        lambda value: Failed("failed") if value == 1 else 1
    )
    failed = sample_law(
        failed_structure,
        _unary_law(failed_unary, failed_variable),
        SamplingOptions(2, 0),
    )
    assert type(failed) is SampledDisproved
    assert failed.undecidable_assignments == 1


def test_inconclusive_evaluation_reason_and_partial_undefined_semantics() -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(
        lambda _value: Indeterminate("bounded")
    )
    inconclusive = sample_law(
        structure, _unary_law(unary, variable), SamplingOptions(2, 0)
    )
    assert type(inconclusive) is SampledInconclusive
    assert inconclusive.reason == "sampled_evaluation_inconclusive"
    assert inconclusive.undecidable_assignments == 2
    assert inconclusive.witness is not None

    partial_structure, _sort, partial_unary, partial_variable, _calls = (
        _unary_structure(
            lambda value: Undefined("outside") if value == 0 else Defined(1),
            partial=True,
        )
    )
    strong = sample_law(
        partial_structure,
        _unary_law(partial_unary, partial_variable),
        SamplingOptions(1, 2),
    )
    conditional = sample_law(
        partial_structure,
        _unary_law(partial_unary, partial_variable, partial="definedness_conditional"),
        SamplingOptions(2, 0),
    )
    assert type(strong) is SampledDisproved and strong.skipped_assignments == 0
    assert type(conditional) is SampledInconclusive
    assert conditional.skipped_assignments == 1
    assert conditional.reason == "sampled_pass_not_proof"


def test_premises_many_sorts_and_zero_variable_domains_preserve_v62_ordering() -> None:
    scalar, vector = Sort("scalar"), Sort("vector")
    scalar_carrier = FiniteCarrier((0, 1), sort=scalar)
    vector_carrier = FiniteCarrier((0, 1), sort=vector)
    structure = (
        StructureBuilder(Signature((scalar, vector)))
        .with_carrier(scalar, scalar_carrier)
        .with_carrier(vector, vector_carrier)
        .freeze()
    )
    x, y = Variable("x", scalar), Variable("y", vector)
    x_term, y_term = Term.variable(x), Term.variable(y)
    many_sorted = Law(
        "many_sorted",
        (x, y),
        Equation(y_term, y_term),
        hypotheses=(Equation(x_term, x_term),),
    )
    report = sample_law(structure, many_sorted, SamplingOptions(4, 0))
    assert type(report) is SampledInconclusive
    assert (
        report.domain_size
        == report.premise_evaluations
        == report.conclusion_evaluations
        == 4
    )

    constant = OperationSymbol("constant", (), scalar)
    zero_structure = (
        StructureBuilder(Signature((scalar,), (constant,)))
        .with_carrier(scalar, scalar_carrier)
        .with_operation(
            constant,
            Operation.from_callable(constant, ((scalar, scalar_carrier),), lambda: 0),
        )
        .freeze()
    )
    constant_term = Term.apply(constant)
    zero_variables = Law("zero", (), Equation(constant_term, constant_term))
    zero = sample_law(zero_structure, zero_variables, SamplingOptions(1, 0))
    assert type(zero) is SampledInconclusive
    assert zero.domain_size == zero.evaluated_samples == 1


def test_premise_partiality_and_comparison_gaps_preserve_short_circuiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_structure, _sort, partial_unary, partial_variable, _calls = (
        _unary_structure(
            lambda value: Undefined("outside") if value == 0 else Defined(1),
            partial=True,
        )
    )
    partial_term = Term.variable(partial_variable)
    partial_premise = Law(
        "partial_premise",
        (partial_variable,),
        Equation(partial_term, partial_term),
        hypotheses=(Equation(Term.apply(partial_unary, partial_term), partial_term),),
    )
    partial = sample_law(partial_structure, partial_premise, SamplingOptions(2, 0))
    assert type(partial) is SampledInconclusive
    assert partial.vacuous_assignments == 1
    assert partial.premise_evaluations == 2 and partial.conclusion_evaluations == 1

    uncertain_structure, _sort, uncertain_unary, uncertain_variable, _calls = (
        _unary_structure(lambda value: Indeterminate("bounded") if value == 1 else 0)
    )
    uncertain_term = Term.variable(uncertain_variable)
    uncertain_premise = Law(
        "uncertain_premise",
        (uncertain_variable,),
        Equation(uncertain_term, uncertain_term),
        hypotheses=(
            Equation(Term.apply(uncertain_unary, uncertain_term), uncertain_term),
        ),
    )
    uncertain = sample_law(
        uncertain_structure, uncertain_premise, SamplingOptions(2, 0)
    )
    assert type(uncertain) is SampledInconclusive
    assert uncertain.undecidable_assignments == 1
    assert uncertain.witness is not None and uncertain.witness.phase == "premise"

    repeated_structure, _sort, repeated_unary, repeated_variable, _calls = (
        _unary_structure(lambda _value: Indeterminate("bounded"))
    )
    repeated_term = Term.variable(repeated_variable)
    repeated_premise = Law(
        "repeated_uncertain_premise",
        (repeated_variable,),
        Equation(repeated_term, repeated_term),
        hypotheses=(
            Equation(Term.apply(repeated_unary, repeated_term), repeated_term),
        ),
    )
    repeated = sample_law(repeated_structure, repeated_premise, SamplingOptions(2, 0))
    assert type(repeated) is SampledInconclusive
    assert repeated.undecidable_assignments == 2

    unequal_structure, _sort, unequal_unary, unequal_variable, _calls = (
        _unary_structure(lambda _value: 0)
    )
    unequal_term = Term.variable(unequal_variable)
    unequal_premise = Law(
        "unequal_premise",
        (unequal_variable,),
        Equation(unequal_term, unequal_term),
        hypotheses=(Equation(Term.apply(unequal_unary, unequal_term), unequal_term),),
    )
    unequal = sample_law(unequal_structure, unequal_premise, SamplingOptions(2, 0))
    assert type(unequal) is SampledInconclusive
    assert unequal.vacuous_assignments == 1

    gap_structure, _sort, gap_unary, gap_variable, _calls = _unary_structure(
        lambda value: value
    )
    gap_term = Term.variable(gap_variable)
    gap_law = Law(
        "comparison_gap",
        (gap_variable,),
        Equation(gap_term, gap_term),
        hypotheses=(Equation(Term.apply(gap_unary, gap_term), gap_term),),
    )
    monkeypatch.setattr(sampling_module, "_equal_defined", lambda *_args: None)
    gap = sample_law(gap_structure, gap_law, SamplingOptions(2, 0))
    assert type(gap) is SampledInconclusive
    assert gap.undecidable_assignments == 2
    assert gap.witness is not None and gap.witness.phase == "premise_comparison"


def test_conclusion_comparison_gap_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(lambda value: value)
    monkeypatch.setattr(sampling_module, "_equal_defined", lambda *_args: None)

    report = sample_law(structure, _unary_law(unary, variable), SamplingOptions(2, 0))

    assert type(report) is SampledInconclusive
    assert report.reason == "sampled_evaluation_inconclusive"
    assert (
        report.witness is not None and report.witness.phase == "conclusion_comparison"
    )


def test_options_preflight_and_domain_bounds_are_typed_and_no_touch() -> None:
    structure, sort, unary, variable, calls = _unary_structure(lambda value: value)
    law = _unary_law(unary, variable)
    for count in (0, -1, True, 65_537):
        with pytest.raises(SamplingDefinitionError, match="sample_count"):
            SamplingOptions(count, 0)
    for seed in (-1, 1 << 64, True):
        with pytest.raises(SamplingDefinitionError, match="seed"):
            SamplingOptions(1, seed)
    assert SamplingOptions(65_536, 0).sample_count == 65_536
    with pytest.raises(SamplingDefinitionError, match="domain size"):
        sample_law(structure, law, SamplingOptions(3, 0))
    assert calls == [0]
    foreign = Variable("foreign", Sort("foreign"))
    foreign_law = Law(
        "foreign", (foreign,), Equation(Term.variable(foreign), Term.variable(foreign))
    )
    with pytest.raises(SamplingDefinitionError, match="variable sort"):
        sample_law(structure, foreign_law, SamplingOptions(1, 0))
    assert calls == [0]
    assert sort is variable.sort


def test_exact_types_sealed_records_hostile_values_and_safe_reprs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure, _sort, unary, variable, _calls = _unary_structure(lambda value: value)
    law = _unary_law(unary, variable)

    class OptionsSubclass(SamplingOptions):
        pass

    with pytest.raises(SamplingDefinitionError, match="exact SamplingOptions"):
        OptionsSubclass(1, 0)
    with pytest.raises(SamplingDefinitionError, match="structure"):
        sample_law(cast(Structure, object()), law, SamplingOptions(1, 0))
    with pytest.raises(SamplingDefinitionError, match="law"):
        sample_law(structure, cast(Law, object()), SamplingOptions(1, 0))
    with pytest.raises(SamplingDefinitionError, match="options"):
        sample_law(structure, law, cast(SamplingOptions, object()))
    report = sample_law(structure, law, SamplingOptions(1, 0))
    other = sample_law(structure, law, SamplingOptions(1, 0))
    assert report is not other and report != other
    for value in (SamplingOptions(1, 0), report):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises((FrozenInstanceError, TypeError)):
            cast(Any, value).seed = 7
        assert type(repr(value)) is str
    with pytest.raises(SamplingDefinitionError, match="sampler-owned"):
        SamplingReport()
    with pytest.raises(SamplingDefinitionError, match="sampler-owned"):
        SampledDisproved()
    with pytest.raises(SamplingDefinitionError, match="sampler-owned"):
        SampledInconclusive()

    class Hostile:
        __hash__ = None  # type: ignore[assignment]

        def __repr__(self) -> str:
            raise RuntimeError("SECRET must not render member")

        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("must not compare member")

    hostile_sort = Sort("hostile")
    hostile = Hostile()
    foreign_hostile = Hostile()
    hostile_structure = (
        StructureBuilder(Signature((hostile_sort,)))
        .with_carrier(hostile_sort, FiniteCarrier((hostile,), sort=hostile_sort))
        .freeze()
    )
    hostile_variable = Variable("hostile", hostile_sort)
    hostile_term = Term.variable(hostile_variable)
    hostile_report = sample_law(
        hostile_structure,
        Law("hostile", (hostile_variable,), Equation(hostile_term, hostile_term)),
        SamplingOptions(1, 0),
    )
    assert type(hostile_report) is SampledInconclusive
    assert "Hostile" not in repr(hostile_report)
    with pytest.raises(TypeError):
        hash(hostile)

    def foreign_defined(*_args: object) -> EvaluationResult:
        return EvaluationResult(Defined(foreign_hostile))

    monkeypatch.setattr(sampling_module, "_evaluate", foreign_defined)
    foreign_report = sample_law(
        hostile_structure,
        Law("foreign", (hostile_variable,), Equation(hostile_term, hostile_term)),
        SamplingOptions(1, 0),
    )
    assert type(foreign_report) is SampledInconclusive
    assert foreign_report.reason == "sampled_evaluation_inconclusive"
    assert foreign_report.witness is not None
    assert foreign_report.witness.phase == "conclusion_comparison"
    assert "SECRET" not in repr(foreign_report)
    assert "SECRET" not in repr(foreign_report.witness)


def test_prng_rejection_path_is_package_owned_and_never_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = iter(((0, (1 << 64) - 1), (0, 1), (0, 2)))
    monkeypatch.setattr(sampling_module, "_next_uint64", lambda _seed: next(state))

    indices = _sample_indices(3, SamplingOptions(2, 0))

    assert indices == (1, 0)
    assert len(set(indices)) == 2
