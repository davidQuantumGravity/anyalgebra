"""Contract tests for immutable operation and relation signature symbols."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.signatures import (
    OperationSymbol,
    RelationSymbol,
    SymbolDefinitionError,
)

SymbolFactory = Callable[[], OperationSymbol | RelationSymbol]
OperationFactory = Callable[..., OperationSymbol]
RelationFactory = Callable[..., RelationSymbol]
operation_factory = cast(OperationFactory, OperationSymbol)
relation_factory = cast(RelationFactory, RelationSymbol)


class _StringSubclass(str):
    """A string subtype used to prove exact built-in-name validation."""


def test_operation_symbol_preserves_ordered_typed_metadata() -> None:
    source = Sort("source")
    target = Sort("target")

    compose = OperationSymbol(
        "compose",
        (source, target, source),
        target,
        notation=";",
    )

    assert compose.name == "compose"
    assert compose.inputs == (source, target, source)
    assert compose.output is target
    assert compose.notation == ";"
    assert compose.arity == 3
    assert (
        repr(compose) == "OperationSymbol(name='compose', inputs=(Sort(name='source'), "
        "Sort(name='target'), Sort(name='source')), "
        "output=Sort(name='target'), notation=';')"
    )


def test_relation_symbol_preserves_ordered_typed_metadata() -> None:
    point = Sort("point")
    line = Sort("line")

    incident = RelationSymbol("incident", (point, line))

    assert incident.name == "incident"
    assert incident.inputs == (point, line)
    assert incident.arity == 2
    assert (
        repr(incident) == "RelationSymbol(name='incident', inputs=(Sort(name='point'), "
        "Sort(name='line')))"
    )


def test_nullary_and_high_arity_symbols_are_valid() -> None:
    scalar = Sort("scalar")
    high_arity_inputs = tuple(Sort(f"s{index}") for index in range(257))

    constant = OperationSymbol("one", (), scalar)
    proposition = RelationSymbol("consistent", ())
    high_arity = OperationSymbol("many", high_arity_inputs, scalar)

    assert constant.arity == 0
    assert constant.inputs == ()
    assert proposition.arity == 0
    assert proposition.inputs == ()
    assert high_arity.arity == 257
    assert high_arity.inputs == high_arity_inputs


def test_inputs_are_one_pass_snapshots_from_generators() -> None:
    left = Sort("left")
    right = Sort("right")
    yielded: list[Sort] = []

    def inputs() -> Iterator[Sort]:
        for sort in (left, right):
            yielded.append(sort)
            yield sort

    symbol = OperationSymbol("pair", inputs(), right)

    assert yielded == [left, right]
    assert symbol.inputs == (left, right)
    assert symbol.arity == 2


def test_inputs_are_independent_of_later_source_mutation() -> None:
    alpha = Sort("alpha")
    beta = Sort("beta")
    supplied = [alpha]

    operation = OperationSymbol("id", supplied, alpha)
    relation = RelationSymbol("selected", supplied)
    supplied.append(beta)

    assert operation.inputs == (alpha,)
    assert relation.inputs == (alpha,)


def test_symbols_are_structural_hashable_ordered_values() -> None:
    left = Sort("left")
    right = Sort("right")
    first = OperationSymbol("pair", (left, right), right, notation=",")
    second = OperationSymbol("pair", [left, right], right, notation=",")
    reversed_inputs = OperationSymbol("pair", (right, left), right, notation=",")
    distinct_notation = OperationSymbol("pair", (left, right), right, notation=";")
    first_relation = RelationSymbol("related", (left, right))
    second_relation = RelationSymbol("related", [left, right])
    reversed_relation = RelationSymbol("related", (right, left))

    assert first == second
    assert hash(first) == hash(second)
    assert first != reversed_inputs
    assert first != distinct_notation
    assert first != cast(object, first_relation)
    assert first_relation == second_relation
    assert hash(first_relation) == hash(second_relation)
    assert first_relation != reversed_relation
    assert {first, second, reversed_inputs} == {first, reversed_inputs}
    assert {first: "operation"}[second] == "operation"


def test_symbols_are_frozen_and_slot_backed() -> None:
    value = Sort("value")
    operation = OperationSymbol("id", (value,), value)
    relation = RelationSymbol("reflexive", (value,))

    with pytest.raises(FrozenInstanceError):
        operation.name = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        relation.inputs = ()  # type: ignore[misc]
    assert not hasattr(operation, "__dict__")
    assert not hasattr(relation, "__dict__")


@pytest.mark.parametrize(
    ("factory", "kind", "field", "reason", "index"),
    [
        (
            lambda: operation_factory("", (), Sort("value")),
            "operation",
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: operation_factory(" leading", (), Sort("value")),
            "operation",
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: relation_factory("trailing ", ()),
            "relation",
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: relation_factory(_StringSubclass("relation"), ()),
            "relation",
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: relation_factory(b"relation", ()),
            "relation",
            "name",
            "must be a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: operation_factory("id", "value", Sort("value")),
            "operation",
            "inputs",
            "must be an iterable of Sort values",
            None,
        ),
        (
            lambda: relation_factory("selected", b"value"),
            "relation",
            "inputs",
            "must be an iterable of Sort values",
            None,
        ),
        (
            lambda: operation_factory("id", [Sort("value"), object()], Sort("value")),
            "operation",
            "inputs",
            "must contain exact Sort values",
            1,
        ),
        (
            lambda: operation_factory("id", (), object()),
            "operation",
            "output",
            "must be an exact Sort",
            None,
        ),
        (
            lambda: operation_factory("id", (), Sort("value"), notation=""),
            "operation",
            "notation",
            "must be None or a non-empty trimmed built-in str",
            None,
        ),
        (
            lambda: operation_factory("id", (), Sort("value"), notation=" spaced "),
            "operation",
            "notation",
            "must be None or a non-empty trimmed built-in str",
            None,
        ),
    ],
)
def test_malformed_symbol_definitions_raise_structured_typed_errors(
    factory: SymbolFactory,
    kind: str,
    field: str,
    reason: str,
    index: int | None,
) -> None:
    with pytest.raises(SymbolDefinitionError) as raised:
        factory()

    error = raised.value
    location = field if index is None else f"{field}[{index}]"
    assert error.kind == kind
    assert error.field == field
    assert error.reason == reason
    assert error.index == index
    assert str(error) == f"invalid {kind} symbol {location}: {reason}"


def test_input_snapshot_normalizes_non_iterable_type_errors() -> None:
    with pytest.raises(SymbolDefinitionError) as raised:
        operation_factory("id", 7, Sort("value"))

    error = raised.value
    assert (error.kind, error.field, error.reason, error.index) == (
        "operation",
        "inputs",
        "could not be snapshotted as an iterable of Sort values",
        None,
    )
    assert isinstance(error.__cause__, TypeError)


def test_input_snapshot_normalizes_generator_runtime_errors() -> None:
    value = Sort("value")

    def failing_inputs() -> Iterator[Sort]:
        yield value
        raise RuntimeError("source iterator failed")

    with pytest.raises(SymbolDefinitionError) as raised:
        relation_factory("reachable", failing_inputs())

    error = raised.value
    assert (error.kind, error.field, error.reason, error.index) == (
        "relation",
        "inputs",
        "could not be snapshotted as an iterable of Sort values",
        None,
    )
    assert isinstance(error.__cause__, RuntimeError)


def test_notation_is_keyword_only_and_part_of_structural_identity() -> None:
    value = Sort("value")
    operation_constructor = cast(Callable[..., OperationSymbol], OperationSymbol)

    plain = OperationSymbol("id", (value,), value)
    displayed = OperationSymbol("id", (value,), value, notation="id")

    assert plain.notation is None
    assert plain != displayed
    with pytest.raises(TypeError):
        operation_constructor("id", (value,), value, "id")


def test_signature_symbols_do_not_prematurely_define_structure_behaviour() -> None:
    value = Sort("value")
    operation = OperationSymbol("multiply", (value, value), value, notation="*")
    relation = RelationSymbol("equal", (value, value))

    for symbol in (operation, relation):
        for attribute in (
            "apply",
            "evaluate",
            "interpret",
            "table",
            "callable",
            "carrier",
            "truth_value",
        ):
            assert not hasattr(symbol, attribute)
