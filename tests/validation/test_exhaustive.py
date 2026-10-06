"""Contract tests for exact bounded exhaustive quantified-law validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.evaluate import EvaluationResult
from anyalgebra.structures.outcomes import Defined, Indeterminate
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.validate import (
    Disproved,
    Inconclusive,
    Proved,
    ValidationDefinitionError,
    ValidationReport,
    ValidationWitness,
    validate_law,
)


def _structure(
    signature: Signature,
    carriers: tuple[FiniteCarrier, ...],
    operations: tuple[Operation | PartialOperation, ...],
) -> Structure:
    builder = StructureBuilder(signature)
    for sort, carrier in zip(signature.sorts, carriers, strict=True):
        builder.with_carrier(sort, carrier)
    for symbol, operation in zip(signature.operations, operations, strict=True):
        builder.with_operation(symbol, operation)
    return builder.freeze()


def _binary_table(
    symbol: OperationSymbol, carrier: FiniteCarrier, function: object
) -> Operation:
    assert callable(function)
    return Operation.from_table(
        symbol,
        ((carrier.sort, carrier),),
        tuple(
            ((left, right), function(left, right))
            for left in carrier
            for right in carrier
        ),
    )


def test_exhaustive_associative_law_proves_and_reports_safe_scope() -> None:
    bit = Sort("bit")
    product = OperationSymbol("and", (bit, bit), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(
        Signature((bit,), (product,)),
        (carrier,),
        (_binary_table(product, carrier, lambda left, right: left & right),),
    )
    x, y, z = (Variable(name, bit) for name in ("x", "y", "z"))
    x_term, y_term, z_term = (Term.variable(value) for value in (x, y, z))
    law = Law(
        "associative",
        (x, y, z),
        Equation(
            Term.apply(product, Term.apply(product, x_term, y_term), z_term),
            Term.apply(product, x_term, Term.apply(product, y_term, z_term)),
        ),
    )

    report = validate_law(structure, law)

    assert type(report) is Proved
    assert (report.expected_assignments, report.evaluated_assignments) == (8, 8)
    assert (report.premise_evaluations, report.conclusion_evaluations) == (0, 8)
    assert report.enumeration_policy == "finite_substitution_domain_lexicographic"
    assert report.variable_carrier_sizes == (2, 2, 2)
    assert (report.algorithm, report.algorithm_version) == (
        "anyalgebra.exhaustive_finite_law",
        1,
    )
    assert report.structure is structure and report.law is law
    assert "0x" not in repr(report) and "Structure(" not in repr(report)
    assert not hasattr(report, "__dict__")
    with pytest.raises(FrozenInstanceError):
        report.expected_assignments = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(report)


def test_first_nonassociative_counterexample_is_deterministic() -> None:
    residue = Sort("residue")
    subtract = OperationSymbol("subtract", (residue, residue), residue)
    carrier = FiniteCarrier((0, 1, 2), sort=residue)
    structure = _structure(
        Signature((residue,), (subtract,)),
        (carrier,),
        (_binary_table(subtract, carrier, lambda left, right: (left - right) % 3),),
    )
    x, y, z = (Variable(name, residue) for name in ("x", "y", "z"))
    x_term, y_term, z_term = (Term.variable(value) for value in (x, y, z))
    law = Law(
        "associative",
        (x, y, z),
        Equation(
            Term.apply(subtract, Term.apply(subtract, x_term, y_term), z_term),
            Term.apply(subtract, x_term, Term.apply(subtract, y_term, z_term)),
        ),
    )

    report = validate_law(structure, law)

    assert type(report) is Disproved
    assert report.witness is not None
    assert report.witness.substitution_indices == (0, 0, 1)
    assert report.evaluated_assignments == 2


def test_premises_are_ordered_vacuous_and_many_sorts_are_exhausted() -> None:
    bit = Sort("bit")
    color = Sort("color")
    bits = FiniteCarrier((0, 1), sort=bit)
    colors = FiniteCarrier(("r", "b"), sort=color)
    structure = _structure(Signature((bit, color)), (bits, colors), ())
    x, y, shade = Variable("x", bit), Variable("y", bit), Variable("shade", color)
    x_term, y_term, shade_term = (Term.variable(item) for item in (x, y, shade))
    conditional = Law(
        "conditional",
        (x, y),
        Equation(x_term, y_term),
        hypotheses=(Equation(x_term, y_term),),
    )
    many_sorted = Law("many_sorted", (x, shade), Equation(shade_term, shade_term))

    conditional_report = validate_law(structure, conditional)
    many_sorted_report = validate_law(structure, many_sorted)

    assert type(conditional_report) is Proved
    assert (
        conditional_report.premise_evaluations,
        conditional_report.conclusion_evaluations,
        conditional_report.vacuous_assignments,
    ) == (4, 2, 2)
    assert type(many_sorted_report) is Proved
    assert many_sorted_report.expected_assignments == 4


def test_partial_undefined_strong_fails_and_conditional_skips() -> None:
    bit = Sort("bit")
    partial_symbol = OperationSymbol("partial", (bit,), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = PartialOperation.from_table(
        partial_symbol, ((bit, carrier),), (), undefined_marker=object()
    )
    structure = _structure(
        Signature((bit,), (partial_symbol,)), (carrier,), (operation,)
    )
    x = Variable("x", bit)
    x_term = Term.variable(x)
    equation = Equation(Term.apply(partial_symbol, x_term), x_term)

    strong = validate_law(structure, Law("strong", (x,), equation))
    conditional = validate_law(
        structure,
        Law("conditional", (x,), equation, partial_semantics="definedness_conditional"),
    )

    assert type(strong) is Disproved and strong.evaluated_assignments == 1
    assert type(conditional) is Proved
    assert (conditional.skipped_assignments, conditional.conclusion_evaluations) == (
        2,
        2,
    )


def test_indeterminate_continues_to_later_definite_disproof() -> None:
    bit = Sort("bit")
    probe = OperationSymbol("probe", (bit,), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = PartialOperation.from_callable(
        probe,
        ((bit, carrier),),
        lambda value: Indeterminate("limit") if value == 0 else Defined(0),
    )
    structure = _structure(Signature((bit,), (probe,)), (carrier,), (operation,))
    x = Variable("x", bit)
    x_term = Term.variable(x)

    report = validate_law(
        structure,
        Law("probe_identity", (x,), Equation(Term.apply(probe, x_term), x_term)),
    )

    assert type(report) is Disproved
    assert report.evaluated_assignments == 2 and report.undecidable_assignments == 1


def test_undefined_premise_is_vacuous_and_undecidable_is_inconclusive() -> None:
    bit = Sort("bit")
    partial = OperationSymbol("partial", (bit,), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    undefined_operation = PartialOperation.from_table(
        partial, ((bit, carrier),), (), undefined_marker=object()
    )
    undefined_structure = _structure(
        Signature((bit,), (partial,)), (carrier,), (undefined_operation,)
    )
    x = Variable("x", bit)
    x_term = Term.variable(x)
    premise_law = Law(
        "undefined_premise",
        (x,),
        Equation(x_term, x_term),
        hypotheses=(Equation(Term.apply(partial, x_term), x_term),),
    )
    vacuous = validate_law(undefined_structure, premise_law)
    assert type(vacuous) is Proved
    assert (vacuous.vacuous_assignments, vacuous.conclusion_evaluations) == (2, 0)

    indeterminate_operation = PartialOperation.from_callable(
        partial, ((bit, carrier),), lambda _value: Indeterminate("limit")
    )
    indeterminate_structure = _structure(
        Signature((bit,), (partial,)), (carrier,), (indeterminate_operation,)
    )
    inconclusive = validate_law(
        indeterminate_structure,
        Law("indeterminate", (x,), Equation(Term.apply(partial, x_term), x_term)),
    )
    assert type(inconclusive) is Inconclusive
    assert inconclusive.witness is not None
    assert inconclusive.witness.substitution_indices == (0,)
    assert "0x" not in repr(inconclusive.witness)


def test_zero_variable_law_and_preflight_type_foreign_sort_errors() -> None:
    bit = Sort("bit")
    zero = OperationSymbol("zero", (), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = Operation.from_table(zero, ((bit, carrier),), (((), 0),))
    structure = _structure(Signature((bit,), (zero,)), (carrier,), (operation,))
    constant = Term.apply(zero)
    zero_variable = validate_law(
        structure, Law("constant", (), Equation(constant, constant))
    )
    assert type(zero_variable) is Proved
    assert (
        zero_variable.expected_assignments,
        zero_variable.evaluated_assignments,
    ) == (1, 1)
    with pytest.raises(ValidationDefinitionError) as wrong_structure:
        validate_law(object(), Law("constant", (), Equation(constant, constant)))  # type: ignore[arg-type]
    assert wrong_structure.value.reason == "must be an exact Structure"
    foreign = Sort("foreign")
    foreign_variable = Variable("foreign", foreign)
    foreign_law = Law(
        "foreign_sort",
        (foreign_variable,),
        Equation(Term.variable(foreign_variable), Term.variable(foreign_variable)),
    )
    with pytest.raises(ValidationDefinitionError) as wrong_sort:
        validate_law(structure, foreign_law)
    assert wrong_sort.value.reason == "variable sort is not declared by structure"


def test_preflight_rejects_foreign_symbols_before_user_callable() -> None:
    bit = Sort("bit")
    called: list[object] = []
    known = OperationSymbol("known", (), bit)
    carrier = FiniteCarrier((0,), sort=bit)
    operation = Operation.from_callable(
        known, ((bit, carrier),), lambda: called.append(1)
    )
    structure = _structure(Signature((bit,), (known,)), (carrier,), (operation,))
    foreign = OperationSymbol("foreign", (), bit)
    law = Law("foreign", (), Equation(Term.apply(foreign), Term.apply(foreign)))

    with pytest.raises(ValidationDefinitionError) as raised:
        validate_law(structure, law)
    assert raised.value.reason == "law operation symbol is not declared by structure"
    assert called == [] and "0x" not in str(raised.value)
    for value in (ValidationReport, ValidationWitness, Proved, Disproved, Inconclusive):
        with pytest.raises(ValidationDefinitionError):
            value()


def test_exact_65536_assignment_bound_needs_no_assignment_storage() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(Signature((bit,)), (carrier,), ())
    variables = tuple(Variable(f"x{index}", bit) for index in range(16))
    term = Term.variable(variables[0])

    report = validate_law(structure, Law("reflexive", variables, Equation(term, term)))

    assert type(report) is Proved
    assert (report.expected_assignments, report.evaluated_assignments) == (
        65_536,
        65_536,
    )


def test_preflight_bounds_and_exact_input_types_are_typed() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(Signature((bit,)), (carrier,), ())
    variables = tuple(Variable(f"x{index}", bit) for index in range(17))
    term = Term.variable(variables[0])
    with pytest.raises(ValidationDefinitionError) as bounds:
        validate_law(structure, Law("too_many", variables, Equation(term, term)))
    assert "Cartesian assignment count" in bounds.value.reason
    with pytest.raises(ValidationDefinitionError) as wrong_law:
        validate_law(structure, object())  # type: ignore[arg-type]
    assert wrong_law.value.reason == "must be an exact Law"
    foreign = Sort("foreign")
    foreign_constant = OperationSymbol("foreign_zero", (), foreign)
    foreign_term = Term.apply(foreign_constant)
    with pytest.raises(ValidationDefinitionError) as result_sort:
        validate_law(
            structure, Law("foreign_result", (), Equation(foreign_term, foreign_term))
        )
    assert (
        result_sort.value.reason == "equation result sort is not declared by structure"
    )


def test_inconclusive_premise_comparison_and_evaluator_failure_are_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(Signature((bit,)), (carrier,), ())
    x = Variable("x", bit)
    term = Term.variable(x)
    law = Law("premise", (x,), Equation(term, term), hypotheses=(Equation(term, term),))

    def indeterminate_evaluator(*_args: object) -> EvaluationResult:
        return EvaluationResult(Indeterminate("limit"), ())

    monkeypatch.setattr(
        "anyalgebra.validation.validate.evaluate_term", indeterminate_evaluator
    )
    premise_report = validate_law(structure, law)
    assert type(premise_report) is Inconclusive
    assert (
        premise_report.witness is not None and premise_report.witness.phase == "premise"
    )

    monkeypatch.setattr(
        "anyalgebra.validation.validate.evaluate_term",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("hidden")),
    )
    failed_report = validate_law(structure, Law("failure", (x,), Equation(term, term)))
    assert type(failed_report) is Inconclusive
    assert failed_report.witness is not None
    assert type(failed_report.witness.left.outcome) is not Defined  # type: ignore[union-attr]


def test_carrier_index_comparison_failure_becomes_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(Signature((bit,)), (carrier,), ())
    x = Variable("x", bit)
    term = Term.variable(x)
    law = Law("comparison", (x,), Equation(term, term))
    monkeypatch.setattr(
        "anyalgebra.validation.validate.evaluate_term",
        lambda *_args: EvaluationResult(Defined(object()), ()),
    )

    report = validate_law(structure, law)

    assert type(report) is Inconclusive
    assert (
        report.witness is not None and report.witness.phase == "conclusion_comparison"
    )
    premise_law = Law(
        "premise_comparison",
        (x,),
        Equation(term, term),
        hypotheses=(Equation(term, term),),
    )
    premise_report = validate_law(structure, premise_law)
    assert type(premise_report) is Inconclusive
    assert (
        premise_report.witness is not None
        and premise_report.witness.phase == "premise_comparison"
    )


def test_stateful_premise_sides_run_once_and_skip_later_work() -> None:
    bit = Sort("bit")
    probe = OperationSymbol("probe", (bit,), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    calls: list[int] = []

    def alternating(_value: object) -> int:
        calls.append(len(calls))
        return len(calls) % 2

    operation = Operation.from_callable(probe, ((bit, carrier),), alternating)
    structure = _structure(Signature((bit,), (probe,)), (carrier,), (operation,))
    x = Variable("x", bit)
    x_term = Term.variable(x)
    probe_term = Term.apply(probe, x_term)
    law = Law(
        "first_premise_false",
        (x,),
        Equation(x_term, x_term),
        hypotheses=(
            Equation(probe_term, probe_term),
            Equation(probe_term, probe_term),
        ),
    )

    report = validate_law(structure, law)

    assert type(report) is Proved
    assert report.premise_evaluations == 2 and report.conclusion_evaluations == 0
    assert calls == [0, 1, 2, 3]


def test_hostile_members_stay_opaque_and_index_comparison_never_uses_raw_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Hostile:
        def __repr__(self) -> str:
            return "secret-member"

        def __eq__(self, other: object) -> bool:
            if self is other:
                return True
            raise RuntimeError("secret equality")

    value = Sort("value")
    stored = Hostile()
    foreign = Hostile()
    carrier = FiniteCarrier((stored,), sort=value)
    structure = _structure(Signature((value,)), (carrier,), ())
    x = Variable("x", value)
    term = Term.variable(x)
    law = Law("opaque", (x,), Equation(term, term))

    proved = validate_law(structure, law)
    assert type(proved) is Proved
    assert "secret-member" not in repr(proved)

    monkeypatch.setattr(
        "anyalgebra.validation.validate.evaluate_term",
        lambda *_args: EvaluationResult(Defined(foreign), ()),
    )
    inconclusive = validate_law(structure, law)
    assert type(inconclusive) is Inconclusive
    assert inconclusive.witness is not None
    assert "secret-member" not in repr(inconclusive.witness)

    class WitnessSubclass(ValidationWitness):
        pass

    with pytest.raises(ValidationDefinitionError):
        WitnessSubclass._create((), "x", None, None, None)


def test_large_balanced_syntax_is_bounded_before_any_operation_execution() -> None:
    bit = Sort("bit")
    combine = OperationSymbol("combine", (bit, bit), bit)
    carrier = FiniteCarrier((0,), sort=bit)
    calls: list[object] = []
    operation = Operation.from_callable(
        combine, ((bit, carrier),), lambda _left, _right: calls.append(1)
    )
    structure = _structure(Signature((bit,), (combine,)), (carrier,), (operation,))
    x = Variable("x", bit)
    node = Term.variable(x)
    for _ in range(16):
        node = Term.apply(combine, node, node)
    law = Law("large", (x,), Equation(node, node))

    with pytest.raises(ValidationDefinitionError) as raised:
        validate_law(structure, law)
    assert raised.value.reason == "term syntax exceeds declared maximum node count"
    assert calls == []
