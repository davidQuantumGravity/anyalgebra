"""Contract tests for immutable many-sorted signatures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.signatures import (
    OperationSymbol,
    RelationSymbol,
    Signature,
    SignatureDefinitionError,
)

SignatureFactory = Callable[..., Signature]
signature_factory = cast(SignatureFactory, Signature)


def test_signature_preserves_declared_many_sorted_syntax() -> None:
    point = Sort("point")
    line = Sort("line")
    incidence = RelationSymbol("incidence", (point, line))
    join = OperationSymbol("join", (point, point), line, notation="join")
    truth = RelationSymbol("true", ())
    origin = OperationSymbol("origin", (), point)

    signature = Signature(
        [point, line],
        operations=(join, origin),
        relations=(incidence, truth),
    )

    assert signature.sorts == (point, line)
    assert signature.sorts[0] is point
    assert signature.sorts[1] is line
    assert signature.operations == (join, origin)
    assert signature.relations == (incidence, truth)
    assert signature.operations[1].arity == 0
    assert signature.relations[1].arity == 0


def test_signature_snapshots_each_declared_collection_once() -> None:
    point = Sort("point")
    line = Sort("line")
    join = OperationSymbol("join", (point, point), line)
    incidence = RelationSymbol("incidence", (point, line))
    yielded: list[str] = []

    def sorts() -> Iterator[Sort]:
        yielded.append("sorts")
        yield point
        yield line

    def operations() -> Iterator[OperationSymbol]:
        yielded.append("operations")
        yield join

    def relations() -> Iterator[RelationSymbol]:
        yielded.append("relations")
        yield incidence

    signature = Signature(sorts(), operations(), relations())

    assert yielded == ["sorts", "operations", "relations"]
    assert signature.sorts == (point, line)
    assert signature.operations == (join,)
    assert signature.relations == (incidence,)


def test_empty_signature_and_source_mutation_boundaries() -> None:
    point = Sort("point")
    origin = OperationSymbol("origin", (), point)
    selected = RelationSymbol("selected", (point,))
    supplied_sorts = [point]
    supplied_operations = [origin]
    supplied_relations = [selected]

    empty = Signature(())
    signature = Signature(supplied_sorts, supplied_operations, supplied_relations)
    supplied_sorts.clear()
    supplied_operations.clear()
    supplied_relations.clear()

    assert empty.sorts == ()
    assert empty.operations == ()
    assert empty.relations == ()
    assert signature.sorts == (point,)
    assert signature.operations == (origin,)
    assert signature.relations == (selected,)


def test_signature_is_frozen_slot_backed_and_structural() -> None:
    point = Sort("point")
    first = Signature((point,), (OperationSymbol("origin", (), point),))
    second = Signature(
        (Sort("point"),), (OperationSymbol("origin", (), Sort("point")),)
    )

    assert first == second
    assert hash(first) == hash(second)
    assert not hasattr(first, "__dict__")
    with pytest.raises(FrozenInstanceError):
        first.sorts = ()  # type: ignore[misc]


def test_signature_allows_shared_operation_and_relation_names() -> None:
    point = Sort("point")
    signature = Signature(
        (point,),
        operations=(OperationSymbol("connected", (point,), point),),
        relations=(RelationSymbol("connected", (point,)),),
    )

    assert signature.operations[0].name == signature.relations[0].name == "connected"


@pytest.mark.parametrize(
    ("factory", "field", "reason", "index", "conflicting_index", "sort_index"),
    [
        (
            lambda: signature_factory("point"),
            "sorts",
            "must be an iterable of exact Sort values",
            None,
            None,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"), object())),
            "sorts",
            "must contain exact Sort values",
            1,
            None,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"), Sort("point"))),
            "sorts",
            "duplicate declared sort",
            1,
            0,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"),), operations="origin"),
            "operations",
            "must be an iterable of exact OperationSymbol values",
            None,
            None,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"),), operations=(object(),)),
            "operations",
            "must contain exact OperationSymbol values",
            0,
            None,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"),), relations=b"related"),
            "relations",
            "must be an iterable of exact RelationSymbol values",
            None,
            None,
            None,
        ),
        (
            lambda: signature_factory((Sort("point"),), relations=(object(),)),
            "relations",
            "must contain exact RelationSymbol values",
            0,
            None,
            None,
        ),
    ],
)
def test_malformed_signature_collections_raise_typed_address_free_diagnostics(
    factory: Callable[[], Signature],
    field: str,
    reason: str,
    index: int | None,
    conflicting_index: int | None,
    sort_index: int | None,
) -> None:
    with pytest.raises(SignatureDefinitionError) as raised:
        factory()

    error = raised.value
    assert error.field == field
    assert error.reason == reason
    assert error.index == index
    assert error.conflicting_index == conflicting_index
    assert error.sort_index == sort_index
    assert "0x" not in str(error)


@pytest.mark.parametrize(
    ("factory", "field", "reason", "index", "conflicting_index"),
    [
        (
            lambda: signature_factory(
                (Sort("point"),),
                operations=(
                    OperationSymbol("origin", (), Sort("point")),
                    OperationSymbol("origin", (Sort("point"),), Sort("point")),
                ),
            ),
            "operations",
            "duplicate operation name",
            1,
            0,
        ),
        (
            lambda: signature_factory(
                (Sort("point"),),
                relations=(
                    RelationSymbol("selected", (Sort("point"),)),
                    RelationSymbol("selected", (Sort("point"), Sort("point"))),
                ),
            ),
            "relations",
            "duplicate relation name",
            1,
            0,
        ),
    ],
)
def test_duplicate_symbol_names_are_rejected_with_declared_indices(
    factory: Callable[[], Signature],
    field: str,
    reason: str,
    index: int,
    conflicting_index: int,
) -> None:
    with pytest.raises(SignatureDefinitionError) as raised:
        factory()

    error = raised.value
    assert (error.field, error.reason, error.index, error.conflicting_index) == (
        field,
        reason,
        index,
        conflicting_index,
    )
    assert error.sort_index is None


@pytest.mark.parametrize(
    ("factory", "field", "index", "sort_index"),
    [
        (
            lambda: signature_factory(
                (Sort("point"),),
                operations=(OperationSymbol("line", (Sort("line"),), Sort("point")),),
            ),
            "operations",
            0,
            0,
        ),
        (
            lambda: signature_factory(
                (Sort("point"),),
                operations=(OperationSymbol("line", (), Sort("line")),),
            ),
            "operations",
            0,
            None,
        ),
        (
            lambda: signature_factory(
                (Sort("point"),),
                relations=(RelationSymbol("on_line", (Sort("point"), Sort("line"))),),
            ),
            "relations",
            0,
            1,
        ),
    ],
)
def test_symbols_cannot_reference_undeclared_sorts(
    factory: Callable[[], Signature], field: str, index: int, sort_index: int | None
) -> None:
    with pytest.raises(SignatureDefinitionError) as raised:
        factory()

    error = raised.value
    assert error.field == field
    assert error.reason == "references an undeclared sort"
    assert error.index == index
    assert error.sort_index == sort_index
    assert error.conflicting_index is None


def test_signature_normalizes_collection_iteration_failures() -> None:
    point = Sort("point")

    def failing_operations() -> Iterator[OperationSymbol]:
        yield OperationSymbol("origin", (), point)
        raise RuntimeError("source iterator failed")

    with pytest.raises(SignatureDefinitionError) as raised:
        Signature((point,), failing_operations())

    error = raised.value
    assert (error.field, error.reason, error.index) == (
        "operations",
        "could not be snapshotted as an iterable of exact OperationSymbol values",
        None,
    )
    assert isinstance(error.__cause__, RuntimeError)


def test_signature_does_not_evaluate_or_interpret_symbols() -> None:
    point = Sort("point")
    signature = Signature(
        (point,),
        operations=(OperationSymbol("origin", (), point),),
        relations=(RelationSymbol("selected", (point,)),),
    )

    for attribute in ("apply", "evaluate", "interpret", "carrier", "table", "callable"):
        assert not hasattr(signature, attribute)
