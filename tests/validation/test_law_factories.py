"""Contract tests for generic, syntax-only quantified-law factories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.factories import (
    LawFactoryError,
    alternative_laws,
    associative_law,
    associativity_law,
    commutative_law,
    commutativity_law,
    custom_law,
    flexible_law,
    flexibility_law,
    jacobi_law,
    jordan_law,
    left_alternativity_law,
    right_alternativity_law,
)


def _binary(name: str, sort: Sort) -> OperationSymbol:
    return OperationSymbol(name, (sort, sort), sort)


def _tree(term: Term) -> str | tuple[OperationSymbol, tuple[object, ...]]:
    """Make every operation node and parenthesization directly comparable."""
    if term.variable_value is not None:
        return term.variable_value.name
    assert term.symbol is not None
    return (term.symbol, tuple(_tree(argument) for argument in term.arguments))


def test_standard_binary_factories_preserve_literal_operation_and_parentheses() -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)

    associative = associative_law(product)
    commutative = commutative_law(product, partial_semantics="definedness_conditional")
    left = left_alternativity_law(product)
    right = right_alternativity_law(product)
    flexible = flexible_law(product)
    jordan = jordan_law(product)

    assert associative.name == "associativity"
    assert tuple(variable.name for variable in associative.variables) == ("x", "y", "z")
    assert associative.conclusion.left.symbol is product
    assert associative.conclusion.left.arguments[0].symbol is product
    assert associative.conclusion.right.symbol is product
    assert associative.conclusion.right.arguments[1].symbol is product
    assert commutative.partial_semantics == "definedness_conditional"
    assert (
        left.conclusion.left.arguments[0].arguments[0]
        is left.conclusion.left.arguments[0].arguments[1]
    )
    assert (
        right.conclusion.right.arguments[1].arguments[0]
        is right.conclusion.right.arguments[1].arguments[1]
    )
    assert (
        flexible.conclusion.left.arguments[0].arguments[0]
        is flexible.conclusion.left.arguments[1]
    )
    assert jordan.conclusion.left.symbol is product
    assert jordan.hypotheses == ()
    assert all(variable.sort is scalar for variable in associative.variables)
    assert alternative_laws(product) == (left, right)
    assert associativity_law is associative_law
    assert commutativity_law is commutative_law
    assert flexibility_law is flexible_law
    assert (
        _tree(associative.conclusion.left),
        _tree(associative.conclusion.right),
    ) == (
        (product, ((product, ("x", "y")), "z")),
        (product, ("x", (product, ("y", "z")))),
    )
    assert (
        _tree(commutative.conclusion.left),
        _tree(commutative.conclusion.right),
    ) == (
        (product, ("x", "y")),
        (product, ("y", "x")),
    )
    assert (_tree(left.conclusion.left), _tree(left.conclusion.right)) == (
        (product, ((product, ("x", "x")), "y")),
        (product, ("x", (product, ("x", "y")))),
    )
    assert (_tree(right.conclusion.left), _tree(right.conclusion.right)) == (
        (product, ((product, ("x", "y")), "y")),
        (product, ("x", (product, ("y", "y")))),
    )
    assert (_tree(flexible.conclusion.left), _tree(flexible.conclusion.right)) == (
        (product, ((product, ("x", "y")), "x")),
        (product, ("x", (product, ("y", "x")))),
    )
    assert (_tree(jordan.conclusion.left), _tree(jordan.conclusion.right)) == (
        (product, ((product, ((product, ("x", "x")), "y")), "x")),
        (product, ((product, ("x", "x")), (product, ("y", "x")))),
    )


def test_jacobi_is_explicit_bracket_addition_and_nullary_zero() -> None:
    scalar = Sort("scalar")
    bracket = _binary("bracket", scalar)
    addition = _binary("addition", scalar)
    zero = OperationSymbol("zero", (), scalar)

    law = jacobi_law(bracket, addition, zero)

    assert law.name == "jacobi"
    assert law.conclusion.left.symbol is addition
    assert law.conclusion.left.arguments[0].symbol is addition
    assert law.conclusion.left.arguments[0].arguments[0].symbol is bracket
    assert law.conclusion.right.symbol is zero
    assert law.conclusion.right.arguments == ()
    assert law.conclusion.left.arguments[0].arguments[0].symbol is bracket
    assert law.conclusion.left.arguments[0].arguments[1].symbol is bracket
    assert law.conclusion.left.arguments[1].symbol is bracket
    assert (_tree(law.conclusion.left), _tree(law.conclusion.right)) == (
        (
            addition,
            (
                (
                    addition,
                    (
                        (bracket, ("x", (bracket, ("y", "z")))),
                        (bracket, ("y", (bracket, ("z", "x")))),
                    ),
                ),
                (bracket, ("z", (bracket, ("x", "y")))),
            ),
        ),
        (zero, ()),
    )


def test_custom_law_uses_explicit_identity_operations_and_many_sorts() -> None:
    source = Sort("source")
    target = Sort("target")
    map_symbol = OperationSymbol("map", (source,), target)
    combine = _binary("combine", target)
    x = Variable("x", source)
    y = Variable("y", target)
    x_term = Term.variable(x)
    y_term = Term.variable(y)
    mapped = Term.apply(map_symbol, x_term)
    premise = Equation(mapped, y_term)
    conclusion = Equation(
        Term.apply(combine, mapped, y_term), Term.apply(combine, y_term, mapped)
    )

    law = custom_law(
        "mapped_commutativity",
        (x, y),
        conclusion,
        operations=(map_symbol, combine),
        premises=(item for item in (premise,)),
        partial_semantics="definedness_conditional",
    )

    assert law.variables == (x, y)
    assert law.conclusion is conclusion
    assert law.hypotheses == (premise,)
    assert law.conclusion.left.symbol is combine
    assert law.conclusion.left.arguments[0].symbol is map_symbol
    assert law.partial_semantics == "definedness_conditional"


def test_factories_reject_incompatible_or_subclass_operation_symbols() -> None:
    scalar = Sort("scalar")
    other = Sort("other")
    unary = OperationSymbol("unary", (scalar,), scalar)
    mixed = OperationSymbol("mixed", (scalar, other), scalar)
    bad_zero = OperationSymbol("bad_zero", (scalar,), scalar)

    class OperationSubclass(OperationSymbol):
        pass

    subclass = cast(OperationSymbol, object.__new__(OperationSubclass))
    for factory, operation, field, reason in (
        (
            associativity_law,
            unary,
            "operation",
            "must be a closed binary OperationSymbol",
        ),
        (
            commutativity_law,
            mixed,
            "operation",
            "must be a closed binary OperationSymbol",
        ),
        (flexibility_law, subclass, "operation", "must be an exact OperationSymbol"),
    ):
        with pytest.raises(LawFactoryError) as raised:
            factory(operation)
        assert (raised.value.field, raised.value.reason) == (field, reason)
        assert "0x" not in str(raised.value)

    with pytest.raises(LawFactoryError) as raised:
        jacobi_law(_binary("bracket", scalar), _binary("add", scalar), bad_zero)
    assert (raised.value.field, raised.value.reason) == (
        "zero",
        "must be a nullary OperationSymbol with the bracket sort output",
    )


def test_custom_law_rejects_one_pass_hostile_and_unlisted_operation_inputs() -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)
    other = _binary("other", scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)
    equation = Equation(Term.apply(product, term, term), term)

    def hostile_operations() -> Iterator[OperationSymbol]:
        yield product
        raise RuntimeError("not public")

    for supplied, field, reason in (
        (
            cast(Iterator[OperationSymbol], "product"),
            "operations",
            "must be an iterable of exact OperationSymbol values",
        ),
        (
            hostile_operations(),
            "operations",
            "could not be snapshotted as an iterable of exact OperationSymbol values",
        ),
        ((other,), "operations", "does not declare an operation used by the law"),
    ):
        with pytest.raises(LawFactoryError) as raised:
            custom_law("custom", (x,), equation, operations=supplied)
        assert (raised.value.field, raised.value.reason) == (field, reason)
        assert "not public" not in str(raised.value)

    equal_looking_but_distinct = _binary("product", scalar)
    with pytest.raises(LawFactoryError) as raised:
        custom_law("custom", (x,), equation, operations=(equal_looking_but_distinct,))
    assert (raised.value.field, raised.value.reason) == (
        "operations",
        "does not declare an operation used by the law",
    )


@pytest.mark.parametrize(
    ("factory", "field", "reason"),
    [
        (
            lambda: custom_law(
                " name",
                (),
                cast(Equation, object()),
                operations=(),
            ),
            "name",
            "must be a non-empty trimmed built-in str",
        ),
        (
            lambda: custom_law(
                "custom",
                cast(Iterator[Variable], "x"),
                cast(Equation, object()),
                operations=(),
            ),
            "variables",
            "must be an iterable of exact Variable values",
        ),
        (
            lambda: custom_law(
                "custom",
                cast(Iterator[Variable], [object()]),
                cast(Equation, object()),
                operations=(),
            ),
            "variables",
            "must contain exact Variable values",
        ),
        (
            lambda: custom_law(
                "custom",
                (Variable("x", Sort("s")), Variable("x", Sort("s"))),
                cast(Equation, object()),
                operations=(),
            ),
            "variables",
            "duplicate quantified variable name",
        ),
        (
            lambda: custom_law("custom", (), cast(Equation, object()), operations=()),
            "conclusion",
            "must be an exact Equation",
        ),
        (
            lambda: custom_law(
                "custom",
                (),
                Equation(
                    Term.variable(Variable("x", Sort("s"))),
                    Term.variable(Variable("x", Sort("s"))),
                ),
                operations=cast(Iterator[OperationSymbol], [object()]),
            ),
            "operations",
            "must contain exact OperationSymbol values",
        ),
        (
            lambda: custom_law(
                "custom",
                (),
                Equation(
                    Term.variable(Variable("x", Sort("s"))),
                    Term.variable(Variable("x", Sort("s"))),
                ),
                operations=cast(Iterator[OperationSymbol], object()),
            ),
            "operations",
            "could not be snapshotted as an iterable of exact OperationSymbol values",
        ),
    ],
)
def test_custom_law_rejects_malformed_declarations(
    factory: object, field: str, reason: str
) -> None:
    assert callable(factory)
    with pytest.raises(LawFactoryError) as raised:
        factory()
    assert (raised.value.field, raised.value.reason) == (field, reason)


def test_custom_law_sanitizes_premises_and_converts_law_definition_failures() -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)
    conclusion = Equation(Term.apply(product, term, term), term)

    def hostile_premises() -> Iterator[Equation]:
        raise RuntimeError("hidden premise")
        yield conclusion

    for premises, reason in (
        (
            cast(Iterator[Equation], "premise"),
            "must be an iterable of exact Equation values",
        ),
        (
            hostile_premises(),
            "could not be snapshotted as an iterable of exact Equation values",
        ),
        (cast(Iterator[Equation], [object()]), "must contain exact Equation values"),
    ):
        with pytest.raises(LawFactoryError) as raised:
            custom_law(
                "custom", (x,), conclusion, operations=(product,), premises=premises
            )
        assert (raised.value.field, raised.value.reason) == ("premises", reason)
        assert "hidden premise" not in str(raised.value)

    undeclared = Variable("undeclared", scalar)
    with pytest.raises(LawFactoryError) as raised:
        custom_law(
            "custom",
            (x,),
            Equation(Term.variable(undeclared), Term.variable(undeclared)),
            operations=(),
        )
    assert (raised.value.field, raised.value.reason) == (
        "conclusion",
        "contains a variable not structurally declared by the quantifier list",
    )


def test_jacobi_rejects_nonexact_zero_and_incompatible_addition() -> None:
    scalar = Sort("scalar")
    other = Sort("other")
    bracket = _binary("bracket", scalar)
    other_addition = _binary("addition", other)

    with pytest.raises(LawFactoryError) as mismatch:
        jacobi_law(bracket, other_addition, OperationSymbol("zero", (), scalar))
    assert (mismatch.value.field, mismatch.value.reason) == (
        "addition",
        "must have the bracket sort as its closed sort",
    )
    with pytest.raises(LawFactoryError) as nonexact:
        jacobi_law(
            bracket, _binary("addition", scalar), cast(OperationSymbol, object())
        )
    assert (nonexact.value.field, nonexact.value.reason) == (
        "zero",
        "must be an exact OperationSymbol",
    )


def test_factories_sanitize_hostile_variables_and_invalid_partial_semantics() -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)

    def hostile_variables() -> Iterator[Variable]:
        yield x
        raise RuntimeError("hidden variable source")

    with pytest.raises(LawFactoryError) as variables_error:
        custom_law(
            "custom",
            hostile_variables(),
            Equation(Term.apply(product, term, term), term),
            operations=(product,),
        )
    assert (variables_error.value.field, variables_error.value.reason) == (
        "variables",
        "could not be snapshotted as an iterable of exact Variable values",
    )
    assert "hidden variable source" not in str(variables_error.value)
    assert variables_error.value.__cause__ is None
    assert variables_error.value.__context__ is None

    with pytest.raises(LawFactoryError) as semantics_error:
        associative_law(product, partial_semantics="unknown")
    assert (semantics_error.value.field, semantics_error.value.reason) == (
        "partial_semantics",
        "must be one of 'strong' or 'definedness_conditional'",
    )


@pytest.mark.parametrize(
    ("argument", "field"),
    [
        ((Variable("x", Sort("s")) for _ in range(513)), "variables"),
        ((OperationSymbol("op", (), Sort("s")) for _ in range(513)), "operations"),
        (
            (
                Equation(
                    Term.variable(Variable("x", Sort("s"))),
                    Term.variable(Variable("x", Sort("s"))),
                )
                for _ in range(513)
            ),
            "premises",
        ),
    ],
)
def test_custom_law_rejects_finite_max_plus_one_declarations(
    argument: object, field: str
) -> None:
    scalar = Sort("scalar")
    x = Variable("x", scalar)
    term = Term.variable(x)
    equation = Equation(term, term)
    kwargs: dict[str, object] = {"operations": ()}
    variables: object = (x,)
    premises: object = ()
    if field == "variables":
        variables = argument
    elif field == "operations":
        kwargs["operations"] = argument
    else:
        premises = argument

    with pytest.raises(LawFactoryError) as raised:
        custom_law(
            "bounded",
            cast(Iterator[Variable], variables),
            equation,
            operations=cast(Iterator[OperationSymbol], kwargs["operations"]),
            premises=cast(Iterator[Equation], premises),
        )
    assert (
        raised.value.field,
        raised.value.reason,
        raised.value.kind,
        raised.value.observed,
        raised.value.maximum,
    ) == (field, "exceeds declared maximum", field, 513, 512)


class _CounterInfinite(Iterator[object]):
    """An inexhaustible declaration source recording the exact pull count."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.pulls = 0

    def __next__(self) -> object:
        self.pulls += 1
        return self.value


@pytest.mark.parametrize("field", ["variables", "operations", "premises"])
def test_custom_law_terminates_infinite_declaration_sources_after_max_plus_one(
    field: str,
) -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)
    equation = Equation(Term.apply(product, term, term), term)
    source = _CounterInfinite(
        x if field == "variables" else product if field == "operations" else equation
    )
    variables: object = (x,)
    operations: object = (product,)
    premises: object = ()
    if field == "variables":
        variables = source
    elif field == "operations":
        operations = source
    else:
        premises = source

    with pytest.raises(LawFactoryError) as raised:
        custom_law(
            "bounded",
            cast(Iterator[Variable], variables),
            equation,
            operations=cast(Iterator[OperationSymbol], operations),
            premises=cast(Iterator[Equation], premises),
        )
    assert raised.value.field == field
    assert source.pulls == 513


def test_custom_law_accepts_exactly_512_variables_operations_and_premises() -> None:
    scalar = Sort("scalar")
    variables = tuple(
        Variable("x", scalar) if index == 0 else Variable(f"v{index}", scalar)
        for index in range(512)
    )
    operations = tuple(
        OperationSymbol(f"identity{index}", (scalar,), scalar) for index in range(512)
    )
    x_term = Term.variable(variables[0])
    conclusion = Equation(Term.apply(operations[0], x_term), x_term)
    premise = Equation(x_term, x_term)

    law = custom_law(
        "at_limit",
        variables,
        conclusion,
        operations=operations,
        premises=(premise,) * 512,
    )

    assert len(law.variables) == len(law.hypotheses) == 512
    assert law.conclusion.left.symbol is operations[0]


def test_custom_law_rejects_duplicate_explicit_operation_identity() -> None:
    scalar = Sort("scalar")
    product = _binary("product", scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)

    with pytest.raises(LawFactoryError) as raised:
        custom_law(
            "duplicate",
            (x,),
            Equation(Term.apply(product, term, term), term),
            operations=(product, product),
        )
    assert (raised.value.field, raised.value.reason, raised.value.index) == (
        "operations",
        "contains a duplicate explicit OperationSymbol declaration",
        1,
    )


def test_custom_law_is_syntax_only_and_law_result_is_sealed_and_immutable() -> None:
    scalar = Sort("scalar")
    identity = OperationSymbol("identity", (scalar,), scalar)
    x = Variable("x", scalar)
    term = Term.variable(x)
    law = custom_law(
        "identity",
        (x,),
        Equation(Term.apply(identity, term), term),
        operations=(identity,),
    )

    assert isinstance(law, Law)
    assert not hasattr(law, "evaluate")
    assert not hasattr(law, "__dict__")
    with pytest.raises(FrozenInstanceError):
        law.name = "changed"  # type: ignore[misc]
