"""Deterministic declared-order counterexample minimization contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import repeat
from typing import Any, cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.evaluate import EvaluationResult
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.sampling import SampledDisproved, SamplingOptions, sample_law
from anyalgebra.validation.validate import Disproved, ValidationWitness, validate_law
from anyalgebra.validation.witness import (
    DeclaredWitnessOrder,
    MinimizedWitness,
    MinimizationInconclusive,
    MinimizationRecord,
    WitnessMinimizationDefinitionError,
    minimize_counterexample,
)
import anyalgebra.validation.witness as witness_module


def _model(
    function: object,
) -> tuple[Structure, OperationSymbol, Variable, Law, list[int]]:
    sort = Sort("scalar")
    carrier = FiniteCarrier((0, 1), sort=sort)
    operation = OperationSymbol("unary", (sort,), sort)
    calls = [0]

    def counted(value: object) -> object:
        calls[0] += 1
        return cast(Any, function)(value)

    structure = (
        StructureBuilder(Signature((sort,), (operation,)))
        .with_carrier(sort, carrier)
        .with_operation(
            operation,
            Operation.from_callable(operation, ((sort, carrier),), counted),
        )
        .freeze()
    )
    variable = Variable("x", sort)
    term = Term.variable(variable)
    return (
        structure,
        operation,
        variable,
        Law("law", (variable,), Equation(Term.apply(operation, term), term)),
        calls,
    )


def test_exhaustive_and_sampled_negative_witnesses_minimize_in_declared_order() -> None:
    structure, _operation, _variable, law, _calls = _model(
        lambda value: 1 - cast(int, value)
    )
    exact = validate_law(structure, law)
    assert type(exact) is Disproved
    lex = DeclaredWitnessOrder.lexicographic(2)
    replay = minimize_counterexample(structure, law, exact, lex)
    assert type(replay) is MinimizedWitness
    assert replay.chosen_assignment_index == 0
    assert replay.source_report is exact
    assert exact.witness is not None and replay.source_witness is not exact.witness
    assert (
        replay.source_witness.substitution_indices,
        replay.source_witness.phase,
        replay.source_witness.premise_index,
        replay.source_witness.left,
        replay.source_witness.right,
    ) == (
        exact.witness.substitution_indices,
        exact.witness.phase,
        exact.witness.premise_index,
        exact.witness.left,
        exact.witness.right,
    )
    sampled = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(sampled) is SampledDisproved
    improved = minimize_counterexample(structure, law, sampled, lex)
    assert type(improved) is MinimizedWitness
    assert sampled.witness is not None and sampled.witness.substitution_indices == (1,)
    assert improved.chosen_assignment_index == 0


def test_reverse_order_and_later_gap_do_not_block_first_definite_counterexample() -> (
    None
):
    structure, _operation, _variable, law, _calls = _model(lambda _value: 0)
    source = validate_law(structure, law)
    assert type(source) is Disproved
    reverse = DeclaredWitnessOrder("reverse", (1, 0), "test.reverse", 1)
    result = minimize_counterexample(structure, law, source, reverse)
    assert type(result) is MinimizedWitness
    assert result.chosen_assignment_index == 1
    assert result.evaluated_indices == (1,)


def test_first_counterexample_stops_before_a_later_indeterminate_branch() -> None:
    structure, _operation, _variable, law, calls = _model(
        lambda value: 0 if value == 1 else Indeterminate("later")
    )
    source = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(source) is SampledDisproved
    calls[0] = 0
    result = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder("reverse", (1, 0), "test", 1)
    )
    assert type(result) is MinimizedWitness
    assert result.chosen_assignment_index == 1
    assert result.evaluated_indices == (1,)
    assert calls == [1]


def test_earlier_gap_blocks_minimality_and_stale_source_is_inconclusive() -> None:
    structure, _operation, _variable, law, _calls = _model(
        lambda value: Indeterminate("bounded") if value == 0 else 0
    )
    source = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(source) is SampledDisproved
    result = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(result) is MinimizationInconclusive
    assert result.reason == "earlier_evaluation_gap"
    assert result.earlier_gap_index == 0

    stale_structure, _operation, _variable, stale_law, _calls = _model(
        lambda value: value
    )
    stale_source = sample_law(stale_structure, stale_law, SamplingOptions(1, 0))
    assert type(stale_source).__name__ == "SampledInconclusive"
    with pytest.raises(WitnessMinimizationDefinitionError, match="negative"):
        minimize_counterexample(
            stale_structure,
            stale_law,
            cast(Any, stale_source),
            DeclaredWitnessOrder.lexicographic(2),
        )


def test_order_preflight_is_bounded_exact_and_no_touch() -> None:
    structure, _operation, _variable, law, calls = _model(lambda _value: 0)
    source = validate_law(structure, law)
    assert type(source) is Disproved
    calls[0] = 0
    for indices in ((0,), (0, 0), (0, 2)):
        with pytest.raises(
            WitnessMinimizationDefinitionError, match="assignment_indices"
        ):
            minimize_counterexample(
                structure, law, source, DeclaredWitnessOrder("bad", indices, "test", 1)
            )
    assert calls == [0]
    with pytest.raises(WitnessMinimizationDefinitionError, match="maximum"):
        DeclaredWitnessOrder("large", repeat(0, 65_537), "test", 1)
    assert DeclaredWitnessOrder.lexicographic(2).assignment_indices == (0, 1)


def test_exact_provenance_immutability_and_safe_records() -> None:
    structure, _operation, _variable, law, _calls = _model(lambda _value: 0)
    source = validate_law(structure, law)
    assert type(source) is Disproved
    order = DeclaredWitnessOrder.lexicographic(2)
    result = minimize_counterexample(structure, law, source, order)
    assert type(result) is MinimizedWitness
    assert result is not minimize_counterexample(structure, law, source, order)
    for value in (order, result):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises((FrozenInstanceError, TypeError)):
            cast(Any, value).name = "changed"
        assert type(repr(value)) is str
    with pytest.raises(WitnessMinimizationDefinitionError, match="minimizer-owned"):
        MinimizedWitness()
    with pytest.raises(WitnessMinimizationDefinitionError, match="minimizer-owned"):
        MinimizationInconclusive()
    with pytest.raises(WitnessMinimizationDefinitionError, match="structure"):
        minimize_counterexample(cast(Structure, object()), law, source, order)
    with pytest.raises(WitnessMinimizationDefinitionError, match="law"):
        minimize_counterexample(structure, cast(Law, object()), source, order)


def test_preflight_snapshots_source_witness_and_order_before_callbacks() -> None:
    holder: dict[str, object] = {}

    def mutating(value: object) -> object:
        if not holder:
            return 0
        source = cast(Disproved, holder["source"])
        order = cast(DeclaredWitnessOrder, holder["order"])
        original = cast(ValidationWitness, holder["witness"])
        object.__setattr__(original, "substitution_indices", (0,))
        object.__setattr__(source, "witness", cast(Any, object()))
        object.__setattr__(order, "name", "mutated")
        object.__setattr__(order, "assignment_indices", (0, 1))
        object.__setattr__(order, "algorithm", "mutated")
        object.__setattr__(order, "algorithm_version", 99)
        return 0

    structure, _operation, _variable, law, _calls = _model(mutating)
    # Arm mutation only after validation has supplied a genuine negative report.
    holder.clear()
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    original_witness = source.witness
    order = DeclaredWitnessOrder("reverse", (1, 0), "test.reverse", 1)
    holder.update(source=source, witness=original_witness, order=order)

    result = minimize_counterexample(structure, law, source, order)

    assert type(result) is MinimizedWitness
    assert result.evaluated_indices == (1,)
    assert result.chosen_assignment_index == 1
    assert result.source_witness is not original_witness
    assert result.source_witness.substitution_indices == (1,)
    assert result.order is not order
    assert (
        result.order.name,
        result.order.assignment_indices,
        result.order.algorithm,
        result.order.algorithm_version,
    ) == ("reverse", (1, 0), "test.reverse", 1)
    assert order.assignment_indices == (0, 1)


def test_stale_negative_replay_and_evaluation_boundary_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"negative": True}
    structure, _operation, variable, law, _calls = _model(
        lambda value: 0 if state["negative"] else value
    )
    source = validate_law(structure, law)
    assert type(source) is Disproved
    state["negative"] = False
    stale = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(stale) is MinimizationInconclusive
    assert stale.reason == "no_current_counterexample"

    state["negative"] = True
    unknown = EvaluationResult(Indeterminate("bounded"))
    monkeypatch.setattr(witness_module, "_evaluate", lambda *_args: unknown)
    gap = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(gap) is MinimizationInconclusive
    assert gap.earlier_gap_witness is not None
    assert gap.earlier_gap_witness.phase == "conclusion"

    monkeypatch.setattr(
        witness_module, "_evaluate", lambda *_args: EvaluationResult(Defined(0))
    )
    monkeypatch.setattr(witness_module, "_equal_defined", lambda *_args: None)
    comparison = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(comparison) is MinimizationInconclusive
    assert comparison.earlier_gap_witness is not None
    assert comparison.earlier_gap_witness.phase == "conclusion_comparison"

    monkeypatch.undo()
    conditional = Law(
        "conditional",
        (variable,),
        law.conclusion,
        partial_semantics="definedness_conditional",
    )
    conditional_source = validate_law(structure, conditional)
    assert type(conditional_source) is Disproved
    undefined = EvaluationResult(Undefined("outside"))
    monkeypatch.setattr(witness_module, "_evaluate", lambda *_args: undefined)
    result = minimize_counterexample(
        structure,
        conditional,
        conditional_source,
        DeclaredWitnessOrder.lexicographic(2),
    )
    assert type(result) is MinimizationInconclusive
    assert result.reason == "no_current_counterexample"


def test_premise_vacuity_and_gaps_and_order_constructor_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure, _operation, variable, law, _calls = _model(lambda _value: 0)
    term = Term.variable(variable)
    premise_law = Law(
        "premise",
        (variable,),
        law.conclusion,
        hypotheses=(Equation(term, term),),
    )
    source = validate_law(structure, premise_law)
    assert type(source) is Disproved
    normal = minimize_counterexample(
        structure, premise_law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(normal) is MinimizedWitness
    undefined = EvaluationResult(Undefined("outside"))
    monkeypatch.setattr(witness_module, "_evaluate", lambda *_args: undefined)
    vacuous = minimize_counterexample(
        structure, premise_law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(vacuous) is MinimizationInconclusive
    assert vacuous.reason == "no_current_counterexample"
    monkeypatch.setattr(
        witness_module,
        "_evaluate",
        lambda *_args: EvaluationResult(Indeterminate("bounded")),
    )
    gap = minimize_counterexample(
        structure, premise_law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(gap) is MinimizationInconclusive
    assert (
        gap.earlier_gap_witness is not None
        and gap.earlier_gap_witness.phase == "premise"
    )
    monkeypatch.setattr(
        witness_module, "_evaluate", lambda *_args: EvaluationResult(Defined(0))
    )
    monkeypatch.setattr(witness_module, "_equal_defined", lambda *_args: None)
    comparison = minimize_counterexample(
        structure, premise_law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(comparison) is MinimizationInconclusive
    assert comparison.earlier_gap_witness is not None
    assert comparison.earlier_gap_witness.phase == "premise_comparison"
    monkeypatch.undo()
    outcomes = iter(
        (
            EvaluationResult(Defined(0)),
            EvaluationResult(Defined(1)),
            EvaluationResult(Defined(0)),
            EvaluationResult(Defined(1)),
        )
    )
    monkeypatch.setattr(witness_module, "_evaluate", lambda *_args: next(outcomes))
    false_premise = minimize_counterexample(
        structure, premise_law, source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(false_premise) is MinimizationInconclusive
    monkeypatch.setattr(
        witness_module,
        "_evaluate",
        lambda *_args: EvaluationResult(Undefined("outside")),
    )
    strong_source = validate_law(structure, law)
    assert type(strong_source) is Disproved
    strong = minimize_counterexample(
        structure, law, strong_source, DeclaredWitnessOrder.lexicographic(2)
    )
    assert type(strong) is MinimizedWitness

    for bad in (
        ("", (0, 1), "test", 1),
        ("name", (0, 1), "", 1),
        ("name", (0, 1), "test", 0),
    ):
        with pytest.raises(WitnessMinimizationDefinitionError):
            DeclaredWitnessOrder(*bad)
    with pytest.raises(WitnessMinimizationDefinitionError, match="could not"):
        DeclaredWitnessOrder("bad", cast(Any, object()), "test", 1)

    class FailingIterator:
        def __iter__(self) -> FailingIterator:
            return self

        def __next__(self) -> int:
            raise RuntimeError("stop")

    with pytest.raises(WitnessMinimizationDefinitionError, match="could not"):
        DeclaredWitnessOrder("bad", FailingIterator(), "test", 1)
    with pytest.raises(WitnessMinimizationDefinitionError, match="non-negative"):
        DeclaredWitnessOrder("bad", (-1,), "test", 1)

    class Child(DeclaredWitnessOrder):
        pass

    with pytest.raises(WitnessMinimizationDefinitionError, match="exact"):
        Child("child", (), "test", 1)
    with pytest.raises(WitnessMinimizationDefinitionError, match="factory"):
        Child.lexicographic(1)
    assert DeclaredWitnessOrder.lexicographic(65_536).assignment_indices[-1] == 65_535
    with pytest.raises(WitnessMinimizationDefinitionError, match="domain_size"):
        DeclaredWitnessOrder.lexicographic(0)
    assert type(repr(DeclaredWitnessOrder.lexicographic(2))) is str
    with pytest.raises(WitnessMinimizationDefinitionError, match="minimizer-owned"):
        MinimizationRecord()
    with pytest.raises(WitnessMinimizationDefinitionError, match="order"):
        minimize_counterexample(
            structure, law, strong_source, cast(DeclaredWitnessOrder, object())
        )
    missing_order = object.__new__(DeclaredWitnessOrder)
    with pytest.raises(WitnessMinimizationDefinitionError, match="missing required"):
        minimize_counterexample(structure, law, strong_source, missing_order)
    malformed_order = DeclaredWitnessOrder.lexicographic(2)
    object.__setattr__(malformed_order, "algorithm_version", 0)
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        minimize_counterexample(structure, law, strong_source, malformed_order)
    other_law = Law("other", (variable,), law.conclusion)
    with pytest.raises(WitnessMinimizationDefinitionError, match="provenance"):
        minimize_counterexample(
            structure, other_law, strong_source, DeclaredWitnessOrder.lexicographic(2)
        )
    forged = object.__new__(Disproved)
    with pytest.raises(WitnessMinimizationDefinitionError, match="missing required"):
        minimize_counterexample(
            structure, law, forged, DeclaredWitnessOrder.lexicographic(2)
        )
    foreign_variable = Variable("foreign", Sort("foreign"))
    foreign_law = Law(
        "foreign",
        (foreign_variable,),
        Equation(Term.variable(foreign_variable), Term.variable(foreign_variable)),
    )
    with pytest.raises(WitnessMinimizationDefinitionError, match="variable sort"):
        witness_module._domain(structure, foreign_law)


def test_tampered_negative_reports_are_rejected_without_operation_calls() -> None:
    structure, _operation, _variable, law, calls = _model(lambda _value: 0)
    domain = witness_module._domain(structure, law)
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    calls[0] = 0
    forged = object.__new__(Disproved)
    with pytest.raises(WitnessMinimizationDefinitionError, match="missing"):
        witness_module._source_coordinates(forged, domain)
    object.__setattr__(source, "witness", cast(Any, object()))
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        witness_module._source_coordinates(source, domain)
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    mixed = ValidationWitness._create(
        source.witness.substitution_indices,
        "conclusion",
        None,
        source.witness.left,
        EvaluationResult(Indeterminate("bounded")),
    )
    object.__setattr__(source, "witness", mixed)
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        witness_module._source_coordinates(source, domain)
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    short = ValidationWitness._create(
        (), "conclusion", None, source.witness.left, source.witness.right
    )
    object.__setattr__(source, "witness", short)
    with pytest.raises(WitnessMinimizationDefinitionError, match="arity"):
        witness_module._source_coordinates(source, domain)
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    out_of_range = ValidationWitness._create(
        (2,), "conclusion", None, source.witness.left, source.witness.right
    )
    object.__setattr__(source, "witness", out_of_range)
    with pytest.raises(WitnessMinimizationDefinitionError, match="range"):
        witness_module._source_coordinates(source, domain)
    source = validate_law(structure, law)
    assert type(source) is Disproved
    object.__setattr__(source, "evaluated_assignments", 1)
    with pytest.raises(WitnessMinimizationDefinitionError, match="match report"):
        witness_module._source_coordinates(source, domain)
    sampled = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(sampled) is SampledDisproved
    object.__setattr__(sampled, "planned_indices", (0,))
    object.__setattr__(sampled, "evaluated_indices", (0,))
    calls[0] = 0
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        witness_module._source_coordinates(sampled, domain)
    sampled = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(sampled) is SampledDisproved
    object.__setattr__(sampled, "seed", 2)
    calls[0] = 0
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        witness_module._source_coordinates(sampled, domain)
    assert calls == [0]
    sampled = sample_law(structure, law, SamplingOptions(1, 0))
    assert type(sampled) is SampledDisproved and sampled.witness is not None
    mismatched = ValidationWitness._create(
        (0,),
        "conclusion",
        None,
        sampled.witness.left,
        sampled.witness.right,
    )
    object.__setattr__(sampled, "witness", mismatched)
    calls[0] = 0
    with pytest.raises(WitnessMinimizationDefinitionError, match="match report"):
        witness_module._source_coordinates(sampled, domain)
    assert calls == [0]


def test_sampled_provenance_tampering_is_strict_and_no_touch() -> None:
    structure, _operation, _variable, law, calls = _model(lambda _value: 0)
    domain = witness_module._domain(structure, law)

    mutations: tuple[tuple[str, object], ...] = (
        ("requested_samples", True),
        ("requested_samples", 0),
        ("requested_samples", 65_537),
        ("requested_samples", 2),
        ("evaluated_samples", True),
        ("evaluated_samples", 0),
        ("evaluated_samples", 2),
        ("seed", True),
        ("seed", -1),
        ("seed", 1 << 64),
        ("planned_indices", [1]),
        ("planned_indices", (True,)),
        ("planned_indices", (2,)),
        ("planned_indices", (1, 1)),
        ("evaluated_indices", [1]),
        ("evaluated_indices", (True,)),
        ("evaluated_indices", (2,)),
        ("evaluated_indices", (1, 1)),
        ("evaluated_indices", (0,)),
        ("domain_size", True),
        ("domain_size", 1),
        ("algorithm", cast(Any, object())),
        ("algorithm", "wrong"),
        ("algorithm_version", True),
        ("algorithm_version", 2),
        ("partial_semantics", cast(Any, object())),
        ("partial_semantics", "definedness_conditional"),
        ("reason", cast(Any, object())),
        ("reason", "wrong"),
    )
    for field, value in mutations:
        sampled = sample_law(structure, law, SamplingOptions(1, 0))
        assert type(sampled) is SampledDisproved
        object.__setattr__(sampled, field, value)
        calls[0] = 0
        with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
            witness_module._source_coordinates(sampled, domain)
        assert calls == [0]

    sampled = sample_law(structure, law, SamplingOptions(2, 0))
    assert type(sampled) is SampledDisproved
    object.__setattr__(sampled, "planned_indices", (0, 1))
    object.__setattr__(sampled, "evaluated_indices", (0,))
    calls[0] = 0
    with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
        witness_module._source_coordinates(sampled, domain)
    assert calls == [0]


def test_source_witness_result_shape_and_v62_strong_partiality() -> None:
    structure, _operation, _variable, law, calls = _model(lambda _value: 0)
    domain = witness_module._domain(structure, law)
    source = validate_law(structure, law)
    assert type(source) is Disproved and source.witness is not None
    for left, right in (
        (EvaluationResult(Defined(0)), EvaluationResult(Indeterminate("gap"))),
        (EvaluationResult(Indeterminate("gap")), EvaluationResult(Defined(0))),
    ):
        forged = ValidationWitness._create((1,), "conclusion", None, left, right)
        object.__setattr__(source, "witness", forged)
        calls[0] = 0
        with pytest.raises(WitnessMinimizationDefinitionError, match="malformed"):
            witness_module._source_coordinates(source, domain)
        assert calls == [0]

    sort = Sort("partial")
    carrier = FiniteCarrier((0,), sort=sort)
    undefined_symbol = OperationSymbol("undefined", (sort,), sort)
    gap_symbol = OperationSymbol("gap", (sort,), sort)
    undefined_operation = PartialOperation.from_callable(
        undefined_symbol,
        ((sort, carrier),),
        lambda _value: Undefined("outside"),
    )
    for gap_outcome in (Indeterminate("gap"), Failed("failed")):
        gap_operation = PartialOperation.from_callable(
            gap_symbol, ((sort, carrier),), lambda _value, outcome=gap_outcome: outcome
        )
        partial_structure = (
            StructureBuilder(Signature((sort,), (undefined_symbol, gap_symbol)))
            .with_carrier(sort, carrier)
            .with_operation(undefined_symbol, undefined_operation)
            .with_operation(gap_symbol, gap_operation)
            .freeze()
        )
        variable = Variable("x", sort)
        term = Term.variable(variable)
        undefined_term = Term.apply(undefined_symbol, term)
        gap_term = Term.apply(gap_symbol, term)
        for conclusion in (
            Equation(undefined_term, gap_term),
            Equation(gap_term, undefined_term),
        ):
            partial_law = Law("strong", (variable,), conclusion)
            partial_source = validate_law(partial_structure, partial_law)
            assert type(partial_source) is Disproved
            result = minimize_counterexample(
                partial_structure,
                partial_law,
                partial_source,
                DeclaredWitnessOrder.lexicographic(1),
            )
            assert type(result) is MinimizedWitness


def test_zero_variable_and_many_sorted_flat_coordinates() -> None:
    scalar = Sort("zero_scalar")
    scalar_carrier = FiniteCarrier((0, 1), sort=scalar)
    left_constant = OperationSymbol("left_constant", (), scalar)
    right_constant = OperationSymbol("right_constant", (), scalar)
    zero_structure = (
        StructureBuilder(Signature((scalar,), (left_constant, right_constant)))
        .with_carrier(scalar, scalar_carrier)
        .with_operation(
            left_constant,
            Operation.from_callable(
                left_constant, ((scalar, scalar_carrier),), lambda: 0
            ),
        )
        .with_operation(
            right_constant,
            Operation.from_callable(
                right_constant, ((scalar, scalar_carrier),), lambda: 1
            ),
        )
        .freeze()
    )
    zero_law = Law(
        "zero",
        (),
        Equation(Term.apply(left_constant), Term.apply(right_constant)),
    )
    zero_source = validate_law(zero_structure, zero_law)
    assert type(zero_source) is Disproved
    zero = minimize_counterexample(
        zero_structure,
        zero_law,
        zero_source,
        DeclaredWitnessOrder.lexicographic(1),
    )
    assert type(zero) is MinimizedWitness
    assert (
        zero.source_assignment_index,
        zero.source_carrier_indices,
        zero.chosen_assignment_index,
        zero.chosen_carrier_indices,
    ) == (0, (), 0, ())

    left_sort, right_sort = Sort("left_sort"), Sort("right_sort")
    left_carrier = FiniteCarrier((0, 1), sort=left_sort)
    right_carrier = FiniteCarrier((10, 11, 12), sort=right_sort)
    binary = OperationSymbol("binary", (left_sort, right_sort), left_sort)
    many_calls = [0]

    def many_function(left: object, right: object) -> object:
        many_calls[0] += 1
        return 0 if (left, right) == (1, 11) else left

    many_structure = (
        StructureBuilder(Signature((left_sort, right_sort), (binary,)))
        .with_carrier(left_sort, left_carrier)
        .with_carrier(right_sort, right_carrier)
        .with_operation(
            binary,
            Operation.from_callable(
                binary,
                ((left_sort, left_carrier), (right_sort, right_carrier)),
                many_function,
            ),
        )
        .freeze()
    )
    left_variable = Variable("left", left_sort)
    right_variable = Variable("right", right_sort)
    left_term, right_term = (
        Term.variable(left_variable),
        Term.variable(right_variable),
    )
    many_law = Law(
        "many",
        (left_variable, right_variable),
        Equation(Term.apply(binary, left_term, right_term), left_term),
    )
    many_source = validate_law(many_structure, many_law)
    assert type(many_source) is Disproved
    many = minimize_counterexample(
        many_structure,
        many_law,
        many_source,
        DeclaredWitnessOrder("chosen", (5, 4, 3, 2, 1, 0), "test", 1),
    )
    assert type(many) is MinimizedWitness
    assert (
        many.source_assignment_index,
        many.source_carrier_indices,
        many.chosen_assignment_index,
        many.chosen_carrier_indices,
        many.evaluated_indices,
    ) == (4, (1, 1), 4, (1, 1), (5, 4))
    assert many_calls[0] == 7


def test_maximum_domain_minimization_stops_after_one_assignment() -> None:
    sort = Sort("wide")
    carrier = FiniteCarrier(tuple(range(256)), sort=sort)
    binary = OperationSymbol("binary", (sort, sort), sort)
    calls = [0]

    def counterexample(_left: object, _right: object) -> object:
        calls[0] += 1
        return 1

    structure = (
        StructureBuilder(Signature((sort,), (binary,)))
        .with_carrier(sort, carrier)
        .with_operation(
            binary,
            Operation.from_callable(binary, ((sort, carrier),), counterexample),
        )
        .freeze()
    )
    left, right = Variable("left", sort), Variable("right", sort)
    left_term, right_term = Term.variable(left), Term.variable(right)
    law = Law(
        "wide",
        (left, right),
        Equation(Term.apply(binary, left_term, right_term), left_term),
    )
    source = validate_law(structure, law)
    assert type(source) is Disproved
    calls[0] = 0
    result = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(65_536)
    )
    assert type(result) is MinimizedWitness
    assert result.domain_size == result.order_size == 65_536
    assert result.evaluated_indices == (0,)
    assert result.chosen_assignment_index == 0
    assert result.chosen_carrier_indices == (0, 0)
    assert calls == [1]


def test_hostile_members_are_opaque_and_order_pull_bound_is_exact() -> None:
    class Hostile:
        repr_calls = 0
        equality_calls = 0

        def __repr__(self) -> str:
            type(self).repr_calls += 1
            raise AssertionError("member repr must stay opaque")

        def __eq__(self, _other: object) -> bool:
            type(self).equality_calls += 1
            raise AssertionError("member equality must stay untouched")

    sort = Sort("hostile")
    member = Hostile()
    carrier = FiniteCarrier((member,), sort=sort)
    partial = OperationSymbol("partial", (sort,), sort)
    operation_calls = [0]

    def undefined(_value: object) -> Undefined:
        operation_calls[0] += 1
        return Undefined("outside")

    structure = (
        StructureBuilder(Signature((sort,), (partial,)))
        .with_carrier(sort, carrier)
        .with_operation(
            partial,
            PartialOperation.from_callable(partial, ((sort, carrier),), undefined),
        )
        .freeze()
    )
    variable = Variable("x", sort)
    term = Term.variable(variable)
    law = Law(
        "hostile",
        (variable,),
        Equation(Term.apply(partial, term), term),
    )
    source = validate_law(structure, law)
    assert type(source) is Disproved
    operation_calls[0] = 0
    with pytest.raises(WitnessMinimizationDefinitionError, match="assignment_indices"):
        minimize_counterexample(
            structure,
            law,
            source,
            DeclaredWitnessOrder("bad", (), "test", 1),
        )
    assert operation_calls == [0]
    assert Hostile.repr_calls == Hostile.equality_calls == 0
    valid = minimize_counterexample(
        structure, law, source, DeclaredWitnessOrder.lexicographic(1)
    )
    assert type(valid) is MinimizedWitness
    assert "Hostile" not in repr(valid)
    assert Hostile.repr_calls == Hostile.equality_calls == 0

    class PullCounter:
        def __init__(self, limit: int | None) -> None:
            self.limit = limit
            self.pulls = 0

        def __iter__(self) -> PullCounter:
            return self

        def __next__(self) -> int:
            self.pulls += 1
            if self.limit is not None and self.pulls > self.limit:
                raise StopIteration
            return 0

    for limit in (None, 65_537):
        counter = PullCounter(limit)
        with pytest.raises(WitnessMinimizationDefinitionError, match="maximum"):
            DeclaredWitnessOrder("large", counter, "test", 1)
        assert counter.pulls == 65_537
    boundary = PullCounter(65_536)
    order = DeclaredWitnessOrder("boundary", boundary, "test", 1)
    assert len(order.assignment_indices) == 65_536
    assert boundary.pulls == 65_537
