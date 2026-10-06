"""Contract tests for standalone recursive typed term evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.evaluate import (
    EvaluationDefinitionError,
    EvaluationResult,
    evaluate_term,
)
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable


def _structure(
    signature: Signature,
    carrier: FiniteCarrier,
    *operations: Operation | PartialOperation,
) -> Structure:
    builder = StructureBuilder(signature).with_carrier(carrier.sort, carrier)
    for operation in operations:
        builder.with_operation(operation.symbol, operation)
    return builder.freeze()


def test_evaluate_preserves_parentheses_and_exact_defined_values() -> None:
    residue = Sort("residue")
    subtract = OperationSymbol("subtract", (residue, residue), residue)
    signature = Signature((residue,), (subtract,))
    carrier = FiniteCarrier((0, 1, 2), sort=residue)
    operation = Operation.from_table(
        subtract,
        [(residue, carrier)],
        [
            ((left, right), (cast(int, left) - cast(int, right)) % 3)
            for left in carrier
            for right in carrier
        ],
    )
    structure = _structure(signature, carrier, operation)
    x_variable = Variable("x", residue)
    y_variable = Variable("y", residue)
    x = Term.variable(x_variable)
    y = Term.variable(y_variable)
    left = Term.apply(subtract, Term.apply(subtract, x, y), y)
    right = Term.apply(subtract, x, Term.apply(subtract, y, y))

    assert evaluate_term(
        structure, left, {Variable("x", residue): 0, y_variable: 1}
    ) == (EvaluationResult(Defined(1), ()))
    assert evaluate_term(
        structure, right, {x_variable: 0, Variable("y", residue): 1}
    ) == (EvaluationResult(Defined(0), ()))


def test_nullary_high_arity_callable_and_partial_operations_evaluate() -> None:
    bit = Sort("bit")
    zero = OperationSymbol("zero", (), bit)
    many = OperationSymbol("many", (bit,) * 257, bit)
    partial_symbol = OperationSymbol("partial", (bit,), bit)
    signature = Signature((bit,), (zero, many, partial_symbol))
    carrier = FiniteCarrier((0, 1), sort=bit)
    constant = Operation.from_callable(zero, [(bit, carrier)], lambda: 1)
    high = Operation.from_callable(many, [(bit, carrier)], lambda *_values: 0)
    partial = PartialOperation.from_table(
        partial_symbol, [(bit, carrier)], [], undefined_marker=object()
    )
    structure = _structure(signature, carrier, constant, high, partial)
    x_variable = Variable("x", bit)
    x = Term.variable(x_variable)

    assert evaluate_term(structure, Term.apply(zero), {}) == EvaluationResult(
        Defined(1), ()
    )
    assert evaluate_term(structure, Term.apply(many, *(x,) * 257), {x_variable: 1}) == (
        EvaluationResult(Defined(0), ())
    )
    undefined = evaluate_term(structure, Term.apply(partial_symbol, x), {x_variable: 0})
    assert undefined.outcome == Undefined("table cell is not declared", witness=(0,))
    assert undefined.path == ()


@pytest.mark.parametrize(
    "outcome",
    [
        Undefined("undefined", witness=(0,)),
        Indeterminate("bound", bounds={"steps": 1}),
        Failed("backend", stage="fixture"),
    ],
)
def test_first_non_defined_child_is_preserved_by_identity_with_child_path(
    outcome: Undefined | Indeterminate | Failed,
) -> None:
    bit = Sort("bit")
    partial_symbol = OperationSymbol("partial", (bit,), bit)
    combine = OperationSymbol("combine", (bit, bit), bit)
    side = OperationSymbol("side", (), bit)
    signature = Signature((bit,), (partial_symbol, combine, side))
    carrier = FiniteCarrier((0, 1), sort=bit)
    partial = PartialOperation.from_callable(
        partial_symbol,
        [(bit, carrier)],
        lambda _value: outcome,
    )

    def combine_function(_left: object, _right: object) -> int:
        raise AssertionError("non-defined child must stop parent application")

    calls: list[str] = []

    def side_function() -> int:
        calls.append("side")
        return 1

    operation = Operation.from_callable(
        combine,
        [(bit, carrier)],
        combine_function,
    )
    side_operation = Operation.from_callable(side, [(bit, carrier)], side_function)
    structure = _structure(signature, carrier, partial, operation, side_operation)
    x_variable = Variable("x", bit)
    x = Term.variable(x_variable)
    nested = Term.apply(combine, Term.apply(partial_symbol, x), Term.apply(side))
    result = evaluate_term(structure, Term.apply(combine, x, nested), {x_variable: 0})

    assert result.outcome is outcome
    assert result.path == (1, 0)
    assert calls == []


def test_environment_validation_is_typed_address_free_and_allows_valid_extras() -> None:
    bit = Sort("bit")
    identity = OperationSymbol("id", (bit,), bit)
    signature = Signature((bit,), (identity,))
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(
        signature,
        carrier,
        Operation.from_table(identity, [(bit, carrier)], [((0,), 0), ((1,), 1)]),
    )
    x_variable = Variable("x", bit)
    x = Term.variable(x_variable)
    extra = Variable("extra", bit)
    assert evaluate_term(structure, x, {x_variable: 1, extra: 0}) == EvaluationResult(
        Defined(1), ()
    )

    cases = [
        (object(), "environment must be a mapping"),
        ({"x": 1}, "environment keys must be exact Variable values"),
        (
            {x_variable: 2},
            "environment value is not a member of its declared carrier",
        ),
        (
            {Variable("z", Sort("other")): 0},
            "environment variable sort is not declared",
        ),
    ]
    for environment, reason in cases:
        with pytest.raises(EvaluationDefinitionError) as caught:
            evaluate_term(structure, x, environment)
        assert caught.value.reason == reason
        assert "0x" not in str(caught.value)
    with pytest.raises(EvaluationDefinitionError) as missing:
        evaluate_term(structure, x, {})
    assert missing.value.reason == "missing required variable binding"


class _BrokenEquality:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_environment_carrier_equality_failure_is_typed_and_address_free() -> None:
    value = Sort("value")
    identity = OperationSymbol("id", (value,), value)
    signature = Signature((value,), (identity,))
    stored = _BrokenEquality()
    carrier = FiniteCarrier((stored,), sort=value)
    structure = _structure(
        signature,
        carrier,
        Operation.from_table(identity, [(value, carrier)], [((stored,), stored)]),
    )
    term = Term.variable(Variable("x", value))

    with pytest.raises(EvaluationDefinitionError) as caught:
        evaluate_term(structure, term, {Variable("x", value): _BrokenEquality()})
    assert caught.value.reason == "environment value carrier equality comparison failed"
    assert "0x" not in str(caught.value)


def test_preflight_rejects_late_missing_bindings_and_symbols_before_callables() -> None:
    bit = Sort("bit")
    side = OperationSymbol("side", (), bit)
    combine = OperationSymbol("combine", (bit, bit), bit)
    signature = Signature((bit,), (side, combine))
    carrier = FiniteCarrier((0, 1), sort=bit)
    calls: list[str] = []

    def side_function() -> int:
        calls.append("side")
        return 0

    structure = _structure(
        signature,
        carrier,
        Operation.from_callable(side, [(bit, carrier)], side_function),
        Operation.from_callable(combine, [(bit, carrier)], lambda _left, _right: 0),
    )
    x = Term.variable(Variable("x", bit))
    missing_later = Term.apply(combine, Term.apply(side), x)
    foreign_later = Term.apply(
        combine, Term.apply(side), Term.apply(OperationSymbol("foreign", (), bit))
    )

    with pytest.raises(EvaluationDefinitionError) as missing:
        evaluate_term(structure, missing_later, {})
    assert missing.value.reason == "missing required variable binding"
    with pytest.raises(EvaluationDefinitionError) as foreign:
        evaluate_term(structure, foreign_later, {})
    assert foreign.value.reason == "term operation symbol is not declared by structure"
    assert calls == []


def test_evaluator_rejects_wrong_inputs_and_foreign_symbols() -> None:
    bit = Sort("bit")
    identity = OperationSymbol("id", (bit,), bit)
    signature = Signature((bit,), (identity,))
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(
        signature,
        carrier,
        Operation.from_table(identity, [(bit, carrier)], [((0,), 0), ((1,), 1)]),
    )
    x_variable = Variable("x", bit)
    x = Term.variable(x_variable)
    equal_reconstructed = OperationSymbol("id", (Sort("bit"),), Sort("bit"))
    equal_term = Term.apply(equal_reconstructed, x)
    absent = Term.apply(OperationSymbol("other", (bit,), bit), x)

    with pytest.raises(EvaluationDefinitionError) as wrong_structure:
        evaluate_term(cast(Structure, object()), x, {})
    assert wrong_structure.value.reason == "structure must be an exact Structure"
    with pytest.raises(EvaluationDefinitionError) as wrong_term:
        evaluate_term(structure, cast(Term, object()), {})
    assert wrong_term.value.reason == "term must be an exact Term"
    assert evaluate_term(structure, equal_term, {x_variable: 0}) == EvaluationResult(
        Defined(0), ()
    )
    with pytest.raises(EvaluationDefinitionError) as unknown:
        evaluate_term(structure, absent, {x_variable: 0})
    assert unknown.value.reason == "term operation symbol is not declared by structure"


def test_evaluation_result_is_frozen_slot_backed_and_validates_path_and_outcome() -> (
    None
):
    result = EvaluationResult(Failed("fixture"), (2, 0))
    assert result.path == (2, 0)
    assert not hasattr(result, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    with pytest.raises(FrozenInstanceError):
        result.path = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact EvaluationOutcome"):
        EvaluationResult(object(), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-negative built-in int"):
        EvaluationResult(Failed("fixture"), (True,))
    with pytest.raises(TypeError, match="Defined evaluation result"):
        EvaluationResult(Defined(1), (0,))
