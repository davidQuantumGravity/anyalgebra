"""Contract tests for typed immutable syntax trees."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol
from anyalgebra.structures.terms import (
    Term,
    TermDefinitionError,
    Variable,
    VariableDefinitionError,
)


class _StringSubclass(str):
    """A string subtype used to prove exact variable-name validation."""


def test_variables_and_terms_preserve_literal_sorts_and_infer_result_sorts() -> None:
    scalar = Sort("scalar")
    vector = Sort("vector")
    x = Variable("x", scalar)
    vector_zero = OperationSymbol("zero", (), vector)
    scale = OperationSymbol("scale", (scalar, vector), vector)

    x_term = Term.variable(x)
    zero = Term.apply(vector_zero)
    scaled = Term.apply(scale, x_term, zero)

    assert x_term.variable_value is x
    assert x_term.symbol is None
    assert x_term.arguments == ()
    assert x_term.sort is scalar
    assert zero.symbol is vector_zero
    assert zero.arguments == ()
    assert zero.sort is vector
    assert scaled.symbol is scale
    assert scaled.arguments == (x_term, zero)
    assert scaled.sort is vector


def test_parentheses_are_immutable_hashable_structural_tree_shape() -> None:
    scalar = Sort("scalar")
    x = Term.variable(Variable("x", scalar))
    y = Term.variable(Variable("y", scalar))
    z = Term.variable(Variable("z", scalar))
    product = OperationSymbol("product", (scalar, scalar), scalar, notation="*")

    left = Term.apply(product, Term.apply(product, x, y), z)
    right = Term.apply(product, x, Term.apply(product, y, z))
    duplicate_left = Term.apply(product, Term.apply(product, x, y), z)

    assert left != right
    assert left == duplicate_left
    assert hash(left) == hash(duplicate_left)
    assert left.arguments[0].arguments == (x, y)
    assert right.arguments[1].arguments == (y, z)
    assert not hasattr(left, "__dict__")
    with pytest.raises(FrozenInstanceError):
        left.sort = scalar  # type: ignore[misc]


def test_nullary_unary_ternary_and_high_arity_operation_terms_are_valid() -> None:
    scalar = Sort("scalar")
    variable = Term.variable(Variable("x", scalar))
    constant = OperationSymbol("one", (), scalar)
    unary = OperationSymbol("negate", (scalar,), scalar)
    ternary = OperationSymbol("choose", (scalar, scalar, scalar), scalar)
    high = OperationSymbol("many", (scalar,) * 257, scalar)

    one = Term.apply(constant)
    negated = Term.apply(unary, one)
    chosen = Term.apply(ternary, variable, one, negated)
    many = Term.apply(high, *(variable,) * 257)

    assert (one.sort, negated.sort, chosen.sort, many.sort) == (scalar,) * 4
    assert many.arguments == (variable,) * 257


def test_malformed_variable_definitions_have_deterministic_typed_diagnostics() -> None:
    with pytest.raises(VariableDefinitionError) as invalid_name:
        Variable(" x", Sort("scalar"))
    with pytest.raises(VariableDefinitionError) as invalid_sort:
        Variable("x", object())  # type: ignore[arg-type]
    with pytest.raises(VariableDefinitionError) as string_subclass:
        Variable(_StringSubclass("x"), Sort("scalar"))

    assert (invalid_name.value.field, invalid_name.value.reason) == (
        "name",
        "must be a non-empty trimmed built-in str",
    )
    assert (invalid_sort.value.field, invalid_sort.value.reason) == (
        "sort",
        "must be an exact Sort",
    )
    assert (string_subclass.value.field, string_subclass.value.reason) == (
        "name",
        "must be a non-empty trimmed built-in str",
    )
    assert "0x" not in str(invalid_name.value)


def test_variables_are_immutable_structural_hashable_values() -> None:
    scalar = Sort("scalar")
    vector = Sort("vector")
    first = Variable("x", scalar)
    duplicate = Variable("x", Sort("scalar"))

    assert first == duplicate
    assert hash(first) == hash(duplicate)
    assert first != Variable("y", scalar)
    assert first != Variable("x", vector)
    assert not hasattr(first, "__dict__")
    with pytest.raises(FrozenInstanceError):
        first.name = "y"  # type: ignore[misc]


def test_term_factories_reject_relation_wrong_arity_and_wrong_argument_types() -> None:
    scalar = Sort("scalar")
    x = Term.variable(Variable("x", scalar))
    binary = OperationSymbol("product", (scalar, scalar), scalar)
    relation = RelationSymbol("equal", (scalar, scalar))

    with pytest.raises(TermDefinitionError) as relation_as_term:
        Term.apply(relation, x, x)  # type: ignore[arg-type]
    with pytest.raises(TermDefinitionError) as wrong_arity:
        Term.apply(binary, x)
    with pytest.raises(TermDefinitionError) as wrong_argument_type:
        Term.apply(binary, x, object())  # type: ignore[arg-type]
    with pytest.raises(TermDefinitionError) as wrong_variable_type:
        Term.variable(object())  # type: ignore[arg-type]

    assert (relation_as_term.value.field, relation_as_term.value.reason) == (
        "symbol",
        "must be an exact OperationSymbol",
    )
    assert (wrong_arity.value.field, wrong_arity.value.reason) == (
        "arguments",
        "expected 2 arguments, got 1",
    )
    assert (wrong_argument_type.value.field, wrong_argument_type.value.index) == (
        "arguments",
        1,
    )
    assert (wrong_variable_type.value.field, wrong_variable_type.value.reason) == (
        "variable",
        "must be an exact Variable",
    )


def test_term_apply_reports_each_ordered_sort_mismatch_without_value_repr() -> None:
    left = Sort("left")
    right = Sort("right")
    left_term = Term.variable(Variable("x", left))
    right_term = Term.variable(Variable("y", right))
    combine = OperationSymbol("combine", (left, right), left)

    with pytest.raises(TermDefinitionError) as raised:
        Term.apply(combine, right_term, left_term)

    error = raised.value
    assert (error.field, error.index, error.reason) == (
        "arguments",
        0,
        "argument sort does not match the operation input sort",
    )
    assert error.expected_sort is left
    assert error.actual_sort is right
    assert str(error) == (
        "invalid term arguments[0]: expected sort 'left', got sort 'right'"
    )
    assert "0x" not in str(error)


def test_term_apply_identifies_a_later_ordered_sort_mismatch() -> None:
    left = Sort("left")
    right = Sort("right")
    left_term = Term.variable(Variable("x", left))
    combine = OperationSymbol("combine", (left, right), left)

    with pytest.raises(TermDefinitionError) as raised:
        Term.apply(combine, left_term, left_term)

    error = raised.value
    assert (error.field, error.index, error.expected_sort, error.actual_sort) == (
        "arguments",
        1,
        right,
        left,
    )


def test_direct_term_construction_cannot_bypass_checked_factories() -> None:
    with pytest.raises(TermDefinitionError) as raised:
        Term()

    assert (raised.value.field, raised.value.reason) == (
        "term",
        "must be constructed with Term.variable() or Term.apply()",
    )


def test_terms_do_not_evaluate_or_assume_a_carrier_or_callable() -> None:
    scalar = Sort("scalar")
    term = Term.variable(Variable("x", scalar))

    for attribute in (
        "evaluate",
        "interpret",
        "carrier",
        "callable",
        "truth_value",
    ):
        assert not hasattr(term, attribute)
