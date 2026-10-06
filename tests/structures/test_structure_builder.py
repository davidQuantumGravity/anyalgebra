"""Contract tests for generic many-sorted structure construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.relations import Relation
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import (
    Structure,
    StructureBuilder,
    StructureDefinitionError,
)


def _flip(symbol: OperationSymbol, carrier: FiniteCarrier) -> Operation:
    return Operation.from_table(
        symbol, [(carrier.sort, carrier)], [((0,), 1), ((1,), 0)]
    )


def test_empty_signature_freezes_to_an_empty_immutable_structure() -> None:
    signature = Signature(())
    builder = StructureBuilder(signature)
    structure = builder.freeze()

    assert structure.signature is signature
    assert structure.carriers == ()
    assert structure.operations == ()
    assert structure.relations == ()
    assert structure == builder.freeze()
    assert structure is not builder.freeze()
    assert not hasattr(structure, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(structure)
    with pytest.raises(FrozenInstanceError):
        structure.signature = signature  # type: ignore[misc]
    assert not hasattr(builder, "__dict__")
    with pytest.raises(AttributeError):
        builder.signature = signature  # type: ignore[misc]


def test_builder_canonicalizes_out_of_order_equal_declarations_and_freezes_order() -> (
    None
):
    bit = Sort("bit")
    flag = Sort("flag")
    flip = OperationSymbol("flip", (bit,), bit)
    choose = OperationSymbol("choose", (bit, flag), flag)
    marked = RelationSymbol("marked", (bit, flag))
    signature = Signature((bit, flag), (flip, choose), (marked,))
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    equal_bit = Sort("bit")
    equal_flag = Sort("flag")
    equal_flip = OperationSymbol("flip", (equal_bit,), equal_bit)
    equal_choose = OperationSymbol("choose", (equal_bit, equal_flag), equal_flag)
    equal_marked = RelationSymbol("marked", (equal_bit, equal_flag))
    partial = PartialOperation.from_table(
        equal_choose,
        [(equal_bit, bits), (equal_flag, flags)],
        [((0, "Y"), "Y")],
        undefined_marker=object(),
    )
    relation = Relation.from_tuples(
        equal_marked, [(equal_bit, bits), (equal_flag, flags)], [(1, "Y")]
    )
    operation = _flip(equal_flip, bits)

    structure = (
        StructureBuilder(signature)
        .with_relation(equal_marked, relation)
        .with_carrier(equal_flag, flags)
        .with_operation(equal_choose, partial)
        .with_carrier(equal_bit, bits)
        .with_operation(equal_flip, operation)
        .freeze()
    )

    assert structure.carriers == (bits, flags)
    assert structure.operations == (operation, partial)
    assert structure.relations == (relation,)


def test_table_callable_partial_and_predicate_interpretations_are_accepted() -> None:
    bit = Sort("bit")
    table_symbol = OperationSymbol("table", (bit,), bit)
    callable_symbol = OperationSymbol("callable", (), bit)
    partial_symbol = OperationSymbol("partial", (bit,), bit)
    predicate_symbol = RelationSymbol("predicate", (bit,))
    signature = Signature(
        (bit,),
        (table_symbol, callable_symbol, partial_symbol),
        (predicate_symbol,),
    )
    carrier = FiniteCarrier((0, 1), sort=bit)
    table = _flip(table_symbol, carrier)
    callable_operation = Operation.from_callable(
        callable_symbol, [(bit, carrier)], lambda: 0
    )
    partial = PartialOperation.from_table(
        partial_symbol, [(bit, carrier)], [((0,), 1)], undefined_marker=object()
    )
    predicate = Relation.from_predicate(
        predicate_symbol, [(bit, carrier)], lambda _value: True
    )

    structure = (
        StructureBuilder(signature)
        .with_carrier(bit, carrier)
        .with_operation(table_symbol, table)
        .with_operation(callable_symbol, callable_operation)
        .with_operation(partial_symbol, partial)
        .with_relation(predicate_symbol, predicate)
        .freeze()
    )
    assert structure.operations == (table, callable_operation, partial)
    assert structure.relations == (predicate,)


def test_missing_unknown_duplicate_and_wrong_type_registrations_are_typed() -> None:
    bit = Sort("bit")
    flip = OperationSymbol("flip", (bit,), bit)
    marked = RelationSymbol("marked", (bit,))
    signature = Signature((bit,), (flip,), (marked,))
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = _flip(flip, carrier)
    relation = Relation.from_tuples(marked, [(bit, carrier)], [(0,)])
    builder = StructureBuilder(signature)

    with pytest.raises(StructureDefinitionError) as wrong_builder:
        StructureBuilder(cast(Signature, object()))
    assert wrong_builder.value.reason == "must be an exact Signature"
    with pytest.raises(StructureDefinitionError) as missing:
        builder.freeze()
    assert missing.value.reason == "missing required carrier registration"
    with pytest.raises(StructureDefinitionError) as wrong_carrier:
        builder.with_carrier(bit, object())
    assert wrong_carrier.value.reason == "carrier must be an exact FiniteCarrier"
    with pytest.raises(StructureDefinitionError) as unknown:
        builder.with_carrier(Sort("other"), carrier)
    assert unknown.value.reason == "unknown declared sort"
    with pytest.raises(StructureDefinitionError) as wrong_sort:
        builder.with_carrier(object(), carrier)
    assert wrong_sort.value.reason == "sort must be an exact Sort"
    wrong_sort_carrier = FiniteCarrier(("N",), sort=Sort("flag"))
    with pytest.raises(StructureDefinitionError) as carrier_sort_mismatch:
        builder.with_carrier(bit, wrong_sort_carrier)
    assert (
        carrier_sort_mismatch.value.reason
        == "carrier sort does not match registration sort"
    )

    builder.with_carrier(bit, carrier)
    with pytest.raises(StructureDefinitionError) as duplicate_carrier:
        builder.with_carrier(bit, carrier)
    assert duplicate_carrier.value.reason == "duplicate carrier registration"
    with pytest.raises(StructureDefinitionError) as wrong_operation:
        builder.with_operation(flip, object())
    assert (
        wrong_operation.value.reason
        == "interpretation must be an Operation or PartialOperation"
    )
    with pytest.raises(StructureDefinitionError) as wrong_relation:
        builder.with_relation(marked, object())
    assert wrong_relation.value.reason == "interpretation must be an exact Relation"
    with pytest.raises(StructureDefinitionError) as wrong_operation_symbol:
        builder.with_operation(object(), operation)
    assert (
        wrong_operation_symbol.value.reason == "symbol must be an exact OperationSymbol"
    )
    with pytest.raises(StructureDefinitionError) as wrong_relation_symbol:
        builder.with_relation(object(), relation)
    assert (
        wrong_relation_symbol.value.reason == "symbol must be an exact RelationSymbol"
    )
    with pytest.raises(StructureDefinitionError) as unknown_operation:
        builder.with_operation(OperationSymbol("other", (), bit), operation)
    assert unknown_operation.value.reason == "unknown declared operation symbol"
    with pytest.raises(StructureDefinitionError) as unknown_relation:
        builder.with_relation(RelationSymbol("other", (bit,)), relation)
    assert unknown_relation.value.reason == "unknown declared relation symbol"
    builder.with_operation(flip, operation).with_relation(marked, relation)
    with pytest.raises(StructureDefinitionError) as duplicate_operation:
        builder.with_operation(flip, operation)
    assert duplicate_operation.value.reason == "duplicate operation registration"
    with pytest.raises(StructureDefinitionError) as duplicate_relation:
        builder.with_relation(marked, relation)
    assert duplicate_relation.value.reason == "duplicate relation registration"


def test_symbol_and_carrier_mismatches_are_rejected_on_registration_or_freeze() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    flip = OperationSymbol("flip", (bit,), bit)
    marked = RelationSymbol("marked", (bit,))
    signature = Signature((bit, flag), (flip,), (marked,))
    bits = FiniteCarrier((0, 1), sort=bit)
    other_bits = FiniteCarrier((0, 2), sort=bit)
    flags = FiniteCarrier(("N",), sort=flag)
    operation = Operation.from_table(flip, [(bit, other_bits)], [((0,), 2), ((2,), 0)])
    relation = Relation.from_tuples(marked, [(bit, other_bits)], [(0,)])
    builder = (
        StructureBuilder(signature).with_carrier(bit, bits).with_carrier(flag, flags)
    )

    with pytest.raises(StructureDefinitionError) as operation_mismatch:
        builder.with_operation(flip, operation)
    assert (
        operation_mismatch.value.reason
        == "interpretation carrier does not match structure carrier"
    )
    with pytest.raises(StructureDefinitionError) as relation_mismatch:
        builder.with_relation(marked, relation)
    assert (
        relation_mismatch.value.reason
        == "interpretation carrier does not match structure carrier"
    )

    wrong_symbol_operation = _flip(OperationSymbol("other", (bit,), bit), bits)
    with pytest.raises(StructureDefinitionError) as symbol_mismatch:
        builder.with_operation(flip, wrong_symbol_operation)
    assert (
        symbol_mismatch.value.reason
        == "interpretation symbol does not match registration"
    )
    deferred = (
        StructureBuilder(Signature((bit,), (flip,)))
        .with_operation(flip, operation)
        .with_carrier(bit, bits)
    )
    with pytest.raises(StructureDefinitionError) as deferred_mismatch:
        deferred.freeze()
    assert (
        deferred_mismatch.value.reason
        == "interpretation carrier does not match structure carrier"
    )


class _BrokenEquality:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_interpretation_carrier_equality_failure_is_normalized() -> None:
    value = Sort("value")
    symbol = OperationSymbol("id", (value,), value)
    signature = Signature((value,), (symbol,))
    structure_value = _BrokenEquality()
    interpretation_value = _BrokenEquality()
    structure_carrier = FiniteCarrier((structure_value,), sort=value)
    interpretation_carrier = FiniteCarrier((interpretation_value,), sort=value)
    operation = Operation.from_table(
        symbol,
        [(value, interpretation_carrier)],
        [((interpretation_value,), interpretation_value)],
    )
    builder = StructureBuilder(signature).with_carrier(value, structure_carrier)

    with pytest.raises(StructureDefinitionError) as caught:
        builder.with_operation(symbol, operation)
    assert caught.value.reason == "interpretation carrier equality comparison failed"


def test_builder_freeze_isolation_and_same_named_structure_coexistence() -> None:
    bit = Sort("bit")
    flip = OperationSymbol("flip", (bit,), bit)
    signature = Signature((bit,), (flip,))
    original = FiniteCarrier((0, 1), sort=bit)
    replacement = FiniteCarrier((0, 2), sort=bit)
    first_operation = _flip(flip, original)
    replacement_operation = Operation.from_table(
        flip, [(bit, replacement)], [((0,), 2), ((2,), 0)]
    )
    builder = (
        StructureBuilder(signature)
        .with_carrier(bit, original)
        .with_operation(flip, first_operation)
    )
    first = builder.freeze()
    second = builder.freeze()
    different = (
        StructureBuilder(signature)
        .with_carrier(bit, replacement)
        .with_operation(flip, replacement_operation)
        .freeze()
    )

    assert first == second
    assert first is not second
    assert first != different
    assert first.carriers == (original,)
    assert different.carriers == (replacement,)


def test_freeze_reports_missing_operation_then_missing_relation() -> None:
    bit = Sort("bit")
    operation_symbol = OperationSymbol("id", (bit,), bit)
    relation_symbol = RelationSymbol("marked", (bit,))
    signature = Signature((bit,), (operation_symbol,), (relation_symbol,))
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = Operation.from_table(
        operation_symbol, [(bit, carrier)], [((0,), 0), ((1,), 1)]
    )
    relation = Relation.from_tuples(relation_symbol, [(bit, carrier)], [(0,)])
    builder = StructureBuilder(signature).with_carrier(bit, carrier)

    with pytest.raises(StructureDefinitionError) as missing_operation:
        builder.freeze()
    assert missing_operation.value.reason == "missing required operation registration"
    builder.with_operation(operation_symbol, operation)
    with pytest.raises(StructureDefinitionError) as missing_relation:
        builder.freeze()
    assert missing_relation.value.reason == "missing required relation registration"
    builder.with_relation(relation_symbol, relation)
    assert builder.freeze().relations == (relation,)


def test_structure_direct_construction_is_rejected() -> None:
    with pytest.raises(StructureDefinitionError) as caught:
        Structure(object(), (), (), ())
    assert caught.value.reason == "use StructureBuilder for construction"
