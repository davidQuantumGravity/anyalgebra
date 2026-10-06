"""Contract tests for finite many-sorted relation evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from inspect import signature
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.relations import (
    Relation,
    RelationApplicationError,
    RelationDefinitionError,
    RelationResult,
)
from anyalgebra.structures.signatures import RelationSymbol


def test_table_relations_return_exact_truth_for_declared_and_absent_tuples() -> None:
    """Extension tuples are true; every other valid carrier tuple is false."""
    person = Sort("person")
    color = Sort("color")
    people = FiniteCarrier((["ada"], ["bert"]), sort=person)
    colors = FiniteCarrier(("blue", "green"), sort=color)
    relation = Relation.from_tuples(
        RelationSymbol("likes", (person, color)),
        [(person, people), (color, colors)],
        [(["ada"], "blue")],
    )

    declared = relation.apply(["ada"], "blue")
    absent = relation.apply(["bert"], "green")
    assert declared == RelationResult(True)
    assert absent == RelationResult(False)
    assert type(declared) is RelationResult
    assert type(declared.holds) is bool
    assert bool(declared) is True
    assert bool(absent) is False


def test_relation_result_requires_an_exact_builtin_bool() -> None:
    """Truth results cannot silently accept integer or arbitrary truthy values."""
    with pytest.raises(TypeError, match="exact built-in bool"):
        RelationResult(1)  # type: ignore[arg-type]

    result = RelationResult(True)
    assert result == RelationResult(True)
    assert hash(result) == hash(RelationResult(True))
    assert type(result) is RelationResult
    assert result != cast(object, True)
    assert result != cast(object, Defined(True))
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.holds = False  # type: ignore[misc]


def test_table_relation_supports_nullary_repeated_many_sorts_and_high_arity() -> None:
    """Canonical extension lookup is independent of arity and hashability."""
    bit = Sort("bit")
    flag = Sort("flag")
    extra = Sort("extra")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    extras = FiniteCarrier(("unused",), sort=extra)
    nullary = Relation.from_tuples(
        RelationSymbol("enabled", ()), [(extra, extras), (bit, bits)], [()]
    )
    repeated = Relation.from_tuples(
        RelationSymbol("different", (bit, bit)), [(bit, bits)], [(0, 1)]
    )
    mixed = Relation.from_tuples(
        RelationSymbol("marked", (bit, flag, bit)),
        [(flag, flags), (bit, bits)],
        [(1, "Y", 0)],
    )
    unit = Sort("unit")
    units = FiniteCarrier((["only"],), sort=unit)
    high = Relation.from_tuples(
        RelationSymbol("high", (unit,) * 257), [(unit, units)], [(["only"],) * 257]
    )

    assert nullary.apply() == RelationResult(True)
    empty_nullary = Relation.from_tuples(
        RelationSymbol("disabled", ()), [(extra, extras)], []
    )
    assert empty_nullary.apply() == RelationResult(False)
    assert repeated.apply(1, 0) == RelationResult(False)
    assert mixed.apply(1, "Y", 0) == RelationResult(True)
    assert high.apply(*(["only"] for _ in range(257))) == RelationResult(True)


def test_relation_sources_are_snapshotted_once_and_mappings_are_coherent() -> None:
    """Bindings and tuples may be generators; mapping keys define extensions."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    bindings = [(bit, carrier)]
    extension = [(0,)]
    yielded_bindings: list[tuple[Sort, FiniteCarrier]] = []
    yielded_tuples: list[tuple[int]] = []

    def binding_source() -> Iterator[tuple[Sort, FiniteCarrier]]:
        for binding in bindings:
            yielded_bindings.append(binding)
            yield binding

    def tuple_source() -> Iterator[tuple[int]]:
        for item in extension:
            yielded_tuples.append(item)
            yield item

    relation = Relation.from_tuples(
        RelationSymbol("selected", (bit,)), binding_source(), tuple_source()
    )
    mapped = Relation.from_tuples(
        RelationSymbol("mapped", (bit,)), {bit: carrier}, {(1,): object()}
    )
    bindings.clear()
    extension.clear()

    assert yielded_bindings == [(bit, carrier)]
    assert yielded_tuples == [(0,)]
    assert relation.carrier_bindings == ((bit, carrier),)
    assert relation.tuples == ((0,),)
    assert mapped.apply(1) == RelationResult(True)


@pytest.mark.parametrize(
    ("tuples", "reason", "entry_index"),
    [
        ([[0]], "input tuple must be an exact tuple", 0),
        ([()], "input tuple has wrong arity", 0),
        ([(2,)], "input is not a member of its declared carrier", 0),
        ([(0,), (0,)], "duplicate relation input tuple", 1),
    ],
)
def test_relation_definition_errors_are_typed_and_address_free(
    tuples: object, reason: str, entry_index: int
) -> None:
    """Malformed extension entries cannot escape as raw lookup failures."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    with pytest.raises(RelationDefinitionError) as caught:
        Relation.from_tuples(
            RelationSymbol("member", (bit,)),
            [(bit, carrier)],
            tuples,  # type: ignore[arg-type]
        )
    assert (caught.value.reason, caught.value.entry_index) == (reason, entry_index)
    assert "0x" not in str(caught.value)


def test_relation_binding_and_application_errors_are_strict() -> None:
    """Construction and application retain sorts without calling a predicate early."""
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N",), sort=flag)
    with pytest.raises(RelationDefinitionError) as missing:
        Relation.from_tuples(RelationSymbol("member", (bit,)), [], [])
    assert missing.value.reason == "missing required carrier binding"
    with pytest.raises(RelationDefinitionError) as duplicate:
        Relation.from_tuples(
            RelationSymbol("member", (bit,)), [(bit, bits), (bit, bits)], []
        )
    assert duplicate.value.reason == "duplicate carrier binding"
    with pytest.raises(RelationDefinitionError) as mismatch:
        Relation.from_tuples(RelationSymbol("member", (bit,)), [(bit, flags)], [])
    assert mismatch.value.reason == "carrier sort does not match binding sort"
    with pytest.raises(RelationDefinitionError) as wrong_symbol:
        Relation.from_tuples(cast(RelationSymbol, object()), [], [])
    assert wrong_symbol.value.reason == "must be an exact RelationSymbol"
    with pytest.raises(RelationDefinitionError) as non_callable:
        Relation.from_predicate(
            RelationSymbol("member", (bit,)),
            [(bit, bits)],
            cast(Callable[..., object], object()),
        )
    assert non_callable.value.reason == "must be callable"

    calls: list[object] = []

    def counting_predicate(value: object) -> bool:
        calls.append(value)
        return True

    relation = Relation.from_predicate(
        RelationSymbol("member", (bit,)), [(bit, bits)], counting_predicate
    )
    with pytest.raises(RelationApplicationError) as arity:
        relation.apply()
    assert (arity.value.expected_arity, arity.value.actual_arity) == (1, 0)
    with pytest.raises(RelationApplicationError) as member:
        relation.apply(3)
    assert member.value.reason == "argument is not a member of its declared carrier"
    assert calls == []


def test_predicate_relations_are_lazy_and_preserve_exact_computational_results() -> (
    None
):
    """Predicates normalize truth but preserve bounded/failure outcomes exactly."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1, 2, 3, 4), sort=bit)
    bounds = {"steps": 5}
    indeterminate = Indeterminate("search limit", bounds=bounds)
    failed = Failed("backend fault", stage="fixture")
    exact = RelationResult(False)
    outputs = {0: True, 1: exact, 2: indeterminate, 3: failed, 4: Undefined("no")}
    calls = 0

    def predicate(value: object) -> object:
        nonlocal calls
        calls += 1
        return outputs[cast(int, value)]

    relation = Relation.from_predicate(
        RelationSymbol("predicate", (bit,)), [(bit, carrier)], predicate, bounds=bounds
    )
    assert tuple(signature(Relation.from_predicate).parameters) == (
        "symbol",
        "carriers",
        "predicate",
        "bounds",
    )
    assert relation.domain == (carrier,)
    assert relation.bounds is bounds
    assert calls == 0
    assert relation.apply(0) == RelationResult(True)
    assert relation.apply(1) is exact
    assert relation.apply(2) is indeterminate
    assert relation.apply(3) is failed
    assert relation.apply(4) == Failed(
        "predicate returned Undefined", stage="predicate"
    )
    assert calls == 5


def test_predicate_relation_failure_translation_is_inert_and_address_free() -> None:
    """Exceptions and malformed predicate branches never become undefinedness."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1, 2), sort=bit)
    malformed = Relation.from_predicate(
        RelationSymbol("bad", (bit,)), [(bit, carrier)], lambda value: value
    )
    raises = Relation.from_predicate(
        RelationSymbol("raise", (bit,)),
        [(bit, carrier)],
        lambda _value: (_ for _ in ()).throw(ValueError("secret detail")),
    )

    assert malformed.apply(1) == Failed(
        "predicate returned invalid relation result", stage="predicate"
    )
    assert raises.apply(0) == Failed("predicate raised ValueError", stage="predicate")

    class FixtureError(Exception):
        """An exception with an invalid mutable runtime type label."""

    FixtureError.__name__ = " unsafe name "
    unsafe = Relation.from_predicate(
        RelationSymbol("unsafe", (bit,)),
        [(bit, carrier)],
        lambda _value: (_ for _ in ()).throw(FixtureError()),
    )
    assert unsafe.apply(0) == Failed("predicate raised exception", stage="predicate")


def test_relation_factory_freezing_hashing_and_predicate_identity() -> None:
    """Tables are structural; private predicates hide their function repr."""
    with pytest.raises(RelationDefinitionError) as direct:
        Relation(object(), (), (), (), (), None)
    assert direct.value.reason == "use Relation.from_tuples for construction"

    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    symbol = RelationSymbol("member", (bit,))
    table = Relation.from_tuples(symbol, [(bit, carrier)], [(0,)])
    equal = Relation.from_tuples(symbol, [(bit, carrier)], [(0,)])
    first = Relation.from_predicate(symbol, [(bit, carrier)], lambda _value: True)
    second = Relation.from_predicate(symbol, [(bit, carrier)], lambda _value: True)

    assert table == equal
    assert first == first
    assert first != second
    assert not hasattr(table, "__dict__")
    assert "<function" not in repr(first)
    for relation in (table, first):
        with pytest.raises(TypeError, match="unhashable"):
            hash(relation)
        with pytest.raises(FrozenInstanceError):
            relation.bounds = None  # type: ignore[misc]


class _BrokenEquality:
    """A value whose non-identical equality comparison cannot be evaluated."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_relation_carrier_equality_failures_remain_typed() -> None:
    """Bad carrier equality is localized at construction and application."""
    value = Sort("value")
    stored = _BrokenEquality()
    carrier = FiniteCarrier((stored,), sort=value)
    with pytest.raises(RelationDefinitionError) as definition:
        Relation.from_tuples(
            RelationSymbol("id", (value,)), [(value, carrier)], [(_BrokenEquality(),)]
        )
    assert definition.value.reason == "input carrier equality comparison failed"

    relation = Relation.from_tuples(
        RelationSymbol("id", (value,)), [(value, carrier)], [(stored,)]
    )
    with pytest.raises(RelationApplicationError) as application:
        relation.apply(_BrokenEquality())
    assert application.value.reason == "argument carrier equality comparison failed"
