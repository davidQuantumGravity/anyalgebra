"""Contract tests for quantified equation and law syntax trees."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from typing import cast
import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.laws import (
    Equation,
    EquationDefinitionError,
    Law,
    LawDefinitionError,
    LawSet,
    LawSetDefinitionError,
)
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable


class _StringSubclass(str):
    """A string subtype used to prove exact built-in-string validation."""


def _equation(sort: Sort, name: str = "x") -> Equation:
    variable = Term.variable(Variable(name, sort))
    return Equation(variable, variable)


def _law(name: str, sort: Sort) -> Law:
    variable = Variable("x", sort)
    term = Term.variable(variable)
    return Law(name, (variable,), Equation(term, term))


def test_equations_preserve_literal_terms_and_require_structurally_equal_sorts() -> (
    None
):
    left_sort = Sort("scalar")
    right_sort = Sort("scalar")
    left = Term.variable(Variable("x", left_sort))
    right = Term.variable(Variable("y", right_sort))

    equation = Equation(left, right)

    assert equation.left is left
    assert equation.right is right
    assert equation.left.sort is left_sort
    assert equation.right.sort is right_sort
    assert equation == Equation(left, right)
    assert hash(equation) == hash(Equation(left, right))
    assert not hasattr(equation, "__dict__")
    with pytest.raises(FrozenInstanceError):
        equation.left = right  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "field", "reason"),
    [
        (
            lambda: Equation(
                cast(Term, object()), Term.variable(Variable("x", Sort("s")))
            ),
            "left",
            "must be an exact Term",
        ),
        (
            lambda: Equation(
                Term.variable(Variable("x", Sort("s"))), cast(Term, object())
            ),
            "right",
            "must be an exact Term",
        ),
    ],
)
def test_equations_reject_nonterm_inputs(
    factory: Callable[[], Equation], field: str, reason: str
) -> None:
    with pytest.raises(EquationDefinitionError) as raised:
        factory()

    error = raised.value
    assert (error.field, error.reason, error.expected_sort, error.actual_sort) == (
        field,
        reason,
        None,
        None,
    )
    assert str(error) == f"invalid equation {field}: {reason}"


def test_equation_rejects_mismatched_sorts_with_structured_metadata() -> None:
    left_sort = Sort("left")
    right_sort = Sort("right")
    left = Term.variable(Variable("x", left_sort))
    right = Term.variable(Variable("y", right_sort))

    with pytest.raises(EquationDefinitionError) as raised:
        Equation(left, right)

    error = raised.value
    assert (error.field, error.reason) == (
        "right",
        "term sort does not match the left term sort",
    )
    assert error.expected_sort is left_sort
    assert error.actual_sort is right_sort
    assert str(error) == (
        "invalid equation right: expected sort 'left', got sort 'right'"
    )
    assert "0x" not in str(error)


def test_law_supports_constant_no_hypothesis_and_arbitrary_operation_arities() -> None:
    scalar = Sort("scalar")
    one = OperationSymbol("one", (), scalar)
    negate = OperationSymbol("negate", (scalar,), scalar)
    choose = OperationSymbol("choose", (scalar, scalar, scalar), scalar)
    many = OperationSymbol("many", (scalar,) * 257, scalar)
    one_term = Term.apply(one)
    law = Law("constant_identity", (), Equation(one_term, one_term))

    x = Term.variable(Variable("x", scalar))
    nested = Term.apply(choose, x, Term.apply(negate, x), one_term)
    high = Term.apply(many, *(nested,) * 257)
    high_law = Law("many_ary", (Variable("x", scalar),), Equation(high, high))

    assert law.variables == ()
    assert law.hypotheses == ()
    assert law.partial_semantics == "strong"
    assert high_law.conclusion.left.arguments == (nested,) * 257


def test_law_preserves_ordered_quantifiers_and_hypotheses_and_is_immutable() -> None:
    scalar = Sort("scalar")
    x = Variable("x", scalar)
    y = Variable("y", scalar)
    x_term = Term.variable(x)
    y_term = Term.variable(y)
    conclusion = Equation(x_term, y_term)
    first_hypothesis = Equation(x_term, x_term)
    second_hypothesis = Equation(y_term, y_term)

    law = Law(
        "ordered",
        [x, y],
        conclusion,
        hypotheses=(hypothesis for hypothesis in (first_hypothesis, second_hypothesis)),
        partial_semantics="definedness_conditional",
    )

    assert law.variables == (x, y)
    assert law.conclusion is conclusion
    assert law.hypotheses == (first_hypothesis, second_hypothesis)
    assert law.partial_semantics == "definedness_conditional"
    assert hash(law) == hash(
        Law(
            "ordered",
            (x, y),
            conclusion,
            hypotheses=(first_hypothesis, second_hypothesis),
            partial_semantics="definedness_conditional",
        )
    )
    assert not hasattr(law, "__dict__")
    with pytest.raises(FrozenInstanceError):
        law.name = "other"  # type: ignore[misc]


def test_law_snapshots_mutable_sources_and_rejects_duplicate_variable_names() -> None:
    scalar = Sort("scalar")
    x = Variable("x", scalar)
    supplied_variables = [x]
    supplied_hypotheses = [_equation(scalar, "x")]
    law = Law(
        "snapshot",
        supplied_variables,
        _equation(scalar, "x"),
        hypotheses=supplied_hypotheses,
    )
    supplied_variables.append(Variable("y", scalar))
    supplied_hypotheses.append(_equation(scalar, "x"))

    assert law.variables == (x,)
    assert len(law.hypotheses) == 1

    with pytest.raises(LawDefinitionError) as raised:
        Law(
            "duplicate",
            (Variable("x", scalar), Variable("x", Sort("other"))),
            _equation(scalar, "x"),
        )

    error = raised.value
    assert (error.field, error.reason, error.index, error.conflicting_index) == (
        "variables",
        "duplicate quantified variable name",
        1,
        0,
    )


@pytest.mark.parametrize(
    ("factory", "field", "reason", "index"),
    [
        (
            lambda: Law(" name", (), _equation(Sort("s"))),
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: Law(_StringSubclass("law"), (), _equation(Sort("s"))),
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: Law("law", cast(Iterator[Variable], "x"), _equation(Sort("s"))),
            "variables",
            "must be an iterable of exact Variable values",
            None,
        ),
        (
            lambda: Law(
                "law", cast(Iterator[Variable], [object()]), _equation(Sort("s"))
            ),
            "variables",
            "must contain exact Variable values",
            0,
        ),
        (
            lambda: Law("law", (), cast(Equation, object())),
            "conclusion",
            "must be an exact Equation",
            None,
        ),
        (
            lambda: Law(
                "law",
                (),
                _equation(Sort("s")),
                hypotheses=cast(Iterator[Equation], "not equations"),
            ),
            "hypotheses",
            "must be an iterable of exact Equation values",
            None,
        ),
        (
            lambda: Law(
                "law",
                (),
                _equation(Sort("s")),
                hypotheses=cast(Iterator[Equation], [object()]),
            ),
            "hypotheses",
            "must contain exact Equation values",
            0,
        ),
        (
            lambda: Law("law", (), _equation(Sort("s")), partial_semantics="weak"),
            "partial_semantics",
            "must be one of 'strong' or 'definedness_conditional'",
            None,
        ),
        (
            lambda: Law(
                "law",
                (),
                _equation(Sort("s")),
                partial_semantics=_StringSubclass("strong"),
            ),
            "partial_semantics",
            "must be one of 'strong' or 'definedness_conditional'",
            None,
        ),
    ],
)
def test_law_rejects_malformed_definition_metadata(
    factory: Callable[[], Law], field: str, reason: str, index: int | None
) -> None:
    with pytest.raises(LawDefinitionError) as raised:
        factory()

    error = raised.value
    assert (error.field, error.reason, error.index) == (field, reason, index)
    assert "0x" not in str(error)


def test_law_rejects_iterators_that_fail_while_snapshotting() -> None:
    scalar = Sort("scalar")

    def failing_variables() -> Iterator[Variable]:
        yield Variable("x", scalar)
        raise RuntimeError("source iterator failed")

    with pytest.raises(LawDefinitionError) as raised:
        Law("failure", failing_variables(), _equation(scalar, "x"))

    assert (raised.value.field, raised.value.reason) == (
        "variables",
        "could not be snapshotted as an iterable of exact Variable values",
    )
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_law_rejects_undeclared_variable_leaves_with_deterministic_paths() -> None:
    scalar = Sort("scalar")
    x = Variable("x", scalar)
    y = Variable("y", scalar)
    pair = OperationSymbol("pair", (scalar, scalar), scalar)
    x_term = Term.variable(x)
    y_term = Term.variable(y)

    with pytest.raises(LawDefinitionError) as conclusion_error:
        Law(
            "undeclared_conclusion",
            (x,),
            Equation(Term.apply(pair, x_term, y_term), x_term),
        )
    with pytest.raises(LawDefinitionError) as hypothesis_error:
        Law(
            "undeclared_hypothesis",
            (x,),
            Equation(x_term, x_term),
            hypotheses=(Equation(x_term, y_term),),
        )

    conclusion = conclusion_error.value
    assert (conclusion.field, conclusion.side, conclusion.term_path) == (
        "conclusion",
        "left",
        (1,),
    )
    hypothesis = hypothesis_error.value
    assert (
        hypothesis.field,
        hypothesis.index,
        hypothesis.side,
        hypothesis.term_path,
    ) == (
        "hypotheses",
        0,
        "right",
        (),
    )
    assert (
        conclusion.reason
        == hypothesis.reason
        == ("contains a variable not structurally declared by the quantifier list")
    )


def test_law_allows_structural_variable_instances_and_unused_quantifiers() -> None:
    scalar = Sort("scalar")
    declared_x = Variable("x", scalar)
    independently_constructed_x = Variable("x", Sort("scalar"))
    unused = Variable("unused", scalar)
    term = Term.variable(independently_constructed_x)

    law = Law("structural_quantifier", (declared_x, unused), Equation(term, term))

    assert law.variables == (declared_x, unused)


def test_law_and_equation_do_not_evaluate_or_claim_relation_atoms() -> None:
    scalar = Sort("scalar")
    law = Law("reflexive", (Variable("x", scalar),), _equation(scalar, "x"))

    for value in (law, law.conclusion):
        for attribute in ("evaluate", "validate", "prove", "disprove", "relation"):
            assert not hasattr(value, attribute)


def test_law_set_is_named_ordered_immutable_and_snapshots_inputs() -> None:
    scalar = Sort("scalar")
    first = _law("first", scalar)
    second = _law("second", scalar)
    supplied = [first, second]
    laws = LawSet("basic", supplied)
    supplied.clear()

    assert laws.name == "basic"
    assert laws.laws == (first, second)
    assert laws == LawSet("basic", (first, second))
    assert hash(laws) == hash(LawSet("basic", (first, second)))
    assert not hasattr(laws, "__dict__")
    with pytest.raises(FrozenInstanceError):
        laws.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "field", "reason", "index"),
    [
        (
            lambda: LawSet("", ()),
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: LawSet(_StringSubclass("set"), ()),
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: LawSet("set", cast(Iterator[Law], "laws")),
            "laws",
            "must be an iterable of exact Law values",
            None,
        ),
        (
            lambda: LawSet("set", cast(Iterator[Law], [object()])),
            "laws",
            "must contain exact Law values",
            0,
        ),
    ],
)
def test_law_set_rejects_malformed_metadata(
    factory: Callable[[], LawSet], field: str, reason: str, index: int | None
) -> None:
    with pytest.raises(LawSetDefinitionError) as raised:
        factory()

    error = raised.value
    assert (error.field, error.reason, error.index) == (field, reason, index)
    assert "0x" not in str(error)


def test_law_set_accepts_empty_and_rejects_duplicate_declared_names() -> None:
    scalar = Sort("scalar")
    first = _law("same", scalar)
    second = _law("same", scalar)

    assert LawSet("empty", ()).laws == ()
    with pytest.raises(LawSetDefinitionError) as raised:
        LawSet("duplicates", (first, second))

    error = raised.value
    assert (error.field, error.reason, error.index, error.conflicting_index) == (
        "laws",
        "duplicate law name",
        1,
        0,
    )


def test_law_set_normalizes_generator_snapshot_failure() -> None:
    scalar = Sort("scalar")

    def failing_laws() -> Iterator[Law]:
        yield _law("first", scalar)
        raise RuntimeError("source iterator failed")

    with pytest.raises(LawSetDefinitionError) as raised:
        LawSet("failing", failing_laws())

    assert (raised.value.field, raised.value.reason) == (
        "laws",
        "could not be snapshotted as an iterable of exact Law values",
    )
    assert isinstance(raised.value.__cause__, RuntimeError)
