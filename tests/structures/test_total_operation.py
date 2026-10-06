"""Contract tests for finite, table-backed total operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import (
    Operation,
    OperationApplicationError,
    OperationDefinitionError,
)
from anyalgebra.structures.operations import TableEntries
from anyalgebra.structures.outcomes import Defined
from anyalgebra.structures.signatures import OperationSymbol


def _carrier_bindings(
    *carriers: FiniteCarrier,
) -> tuple[tuple[Sort, FiniteCarrier], ...]:
    return tuple((carrier.sort, carrier) for carrier in carriers)


class _BrokenEquality:
    """A value that makes non-identical finite-carrier equality unavailable."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_nullary_unary_and_many_sorted_ternary_tables_apply_exactly() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)

    zero = Operation.from_table(
        OperationSymbol("zero", (), bit), _carrier_bindings(bits), [((), 0)]
    )
    flip = Operation.from_table(
        OperationSymbol("flip", (bit,), bit),
        _carrier_bindings(bits),
        [((0,), 1), ((1,), 0)],
    )
    choose = Operation.from_table(
        OperationSymbol("choose", (bit, flag, bit), flag),
        _carrier_bindings(bits, flags),
        [
            ((left, selected, right), "Y" if selected == "Y" else "N")
            for left in bits
            for selected in flags
            for right in bits
        ],
    )

    assert zero.apply() == Defined(0)
    assert flip.apply(0) == Defined(1)
    assert choose.apply(1, "Y", 0) == Defined("Y")
    assert choose.symbol.arity == 3
    assert choose.input_carriers == (bits, flags, bits)
    assert choose.output_carrier is flags


def test_repeated_sort_and_257_ary_singleton_table_are_supported() -> None:
    unit_sort = Sort("unit")
    unit = FiniteCarrier(("only",), sort=unit_sort)
    high = OperationSymbol("high", (unit_sort,) * 257, unit_sort)
    operation = Operation.from_table(
        high,
        _carrier_bindings(unit),
        [(tuple("only" for _ in range(257)), "only")],
    )

    assert operation.apply(*(["only"] * 257)) == Defined("only")


def test_unhashable_carrier_items_are_canonicalized_without_hashing() -> None:
    list_sort = Sort("list")
    carrier = FiniteCarrier(([0], [1]), sort=list_sort)
    operation = Operation.from_table(
        OperationSymbol("right", (list_sort, list_sort), list_sort),
        _carrier_bindings(carrier),
        [
            (([0], [0]), [0]),
            (([0], [1]), [1]),
            (([1], [0]), [0]),
            (([1], [1]), [1]),
        ],
    )

    assert operation.apply([1], [0]) == Defined([0])


def test_sources_are_snapshotted_once_and_isolated_from_container_mutation() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    bindings = [(bit, carrier)]
    entries = [((0,), 1), ((1,), 0)]
    yielded_bindings: list[tuple[Sort, FiniteCarrier]] = []
    yielded: list[tuple[tuple[int], int]] = []

    def carrier_bindings() -> Iterator[tuple[Sort, FiniteCarrier]]:
        for binding in bindings:
            yielded_bindings.append(binding)
            yield binding

    def table() -> Iterator[tuple[tuple[int], int]]:
        for entry in entries:
            yielded.append(entry)
            yield entry

    operation = Operation.from_table(
        OperationSymbol("flip", (bit,), bit), carrier_bindings(), table()
    )
    bindings.clear()
    entries[:] = [((0,), 0), ((1,), 1)]

    assert yielded_bindings == [(bit, carrier)]
    assert yielded == [((0,), 1), ((1,), 0)]
    assert operation.apply(0) == Defined(1)
    assert operation.carrier_bindings == ((bit, carrier),)


@pytest.mark.parametrize(
    ("table", "field", "reason", "entry_index"),
    [
        ([(0,)], "table", "entry must be a pair", 0),
        ([([0], 1), ((1,), 0)], "table", "input key must be an exact tuple", 0),
        ([((0, 1), 0), ((1, 0), 1)], "table", "input tuple has wrong arity", 0),
        (
            [((2,), 0), ((0,), 0), ((1,), 1)],
            "table",
            "input is not a member of its declared carrier",
            0,
        ),
        (
            [((0,), 2), ((1,), 1)],
            "table",
            "output is not a member of its declared carrier",
            0,
        ),
    ],
)
def test_malformed_table_entries_raise_structured_definition_errors(
    table: object, field: str, reason: str, entry_index: int
) -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)

    with pytest.raises(OperationDefinitionError) as caught:
        Operation.from_table(
            OperationSymbol("id", (bit,), bit),
            _carrier_bindings(carrier),
            cast(TableEntries, table),
        )

    error = caught.value
    assert isinstance(error, AnyAlgebraError)
    assert (error.field, error.reason, error.entry_index) == (
        field,
        reason,
        entry_index,
    )
    assert "0x" not in str(error)


def test_duplicate_and_missing_cells_have_deterministic_coordinates() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    symbol = OperationSymbol("id", (bit,), bit)

    with pytest.raises(OperationDefinitionError) as duplicate:
        Operation.from_table(
            symbol,
            _carrier_bindings(carrier),
            [((0,), 0), ((0,), 0), ((1,), 1)],
        )
    assert duplicate.value.reason == "duplicate table input tuple"
    assert duplicate.value.entry_index == 1
    assert duplicate.value.conflicting_entry_index == 0

    with pytest.raises(OperationDefinitionError) as missing:
        Operation.from_table(symbol, _carrier_bindings(carrier), [((1,), 1)])
    assert missing.value.reason == "table is missing required input tuple"
    assert missing.value.index_tuple == (0,)


def test_carrier_binding_errors_are_deterministic_and_do_not_render_values() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    symbol = OperationSymbol("choose", (bit, flag), bit)

    cases = [
        ([(bit, bits)], "missing required carrier binding", None),
        ([(bit, bits), (bit, bits), (flag, flags)], "duplicate carrier binding", 1),
        ([(bit, flags), (flag, flags)], "carrier sort does not match binding sort", 0),
        (
            [(bit, object()), (flag, flags)],
            "binding must contain an exact FiniteCarrier",
            0,
        ),
    ]
    for bindings, reason, binding_index in cases:
        with pytest.raises(OperationDefinitionError) as caught:
            Operation.from_table(
                symbol,
                bindings,  # type: ignore[arg-type]
                [((left, right), left) for left in bits for right in flags],
            )
        assert caught.value.reason == reason
        assert caught.value.binding_index == binding_index
        assert "object at" not in str(caught.value)


def test_extra_valid_carrier_bindings_are_retained_without_affecting_operation() -> (
    None
):
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    operation = Operation.from_table(
        OperationSymbol("flip", (bit,), bit),
        [(flag, flags), (bit, bits)],
        [((0,), 1), ((1,), 0)],
    )

    assert operation.carrier_bindings == ((flag, flags), (bit, bits))
    assert operation.input_carriers == (bits,)
    assert operation.output_carrier is bits
    assert operation.apply(0) == Defined(1)


def test_apply_rejects_invalid_requests_before_lookup() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = Operation.from_table(
        OperationSymbol("flip", (bit,), bit),
        _carrier_bindings(carrier),
        [((0,), 1), ((1,), 0)],
    )

    with pytest.raises(OperationApplicationError) as wrong_arity:
        operation.apply()
    assert wrong_arity.value.reason == "wrong number of arguments"
    assert (wrong_arity.value.expected_arity, wrong_arity.value.actual_arity) == (1, 0)

    with pytest.raises(OperationApplicationError) as wrong_member:
        operation.apply(3)
    assert (
        wrong_member.value.reason == "argument is not a member of its declared carrier"
    )
    assert wrong_member.value.argument_index == 0
    assert wrong_member.value.expected_sort == bit


def test_carrier_equality_failure_is_a_typed_application_error() -> None:
    value = Sort("value")
    stored = _BrokenEquality()
    carrier = FiniteCarrier((stored,), sort=value)
    operation = Operation.from_table(
        OperationSymbol("id", (value,), value),
        _carrier_bindings(carrier),
        [((stored,), stored)],
    )

    with pytest.raises(OperationApplicationError) as caught:
        operation.apply(_BrokenEquality())

    assert caught.value.reason == "argument carrier equality comparison failed"
    assert caught.value.argument_index == 0


def test_output_carrier_equality_failure_is_a_typed_definition_error() -> None:
    value = Sort("value")
    stored = _BrokenEquality()
    carrier = FiniteCarrier((stored,), sort=value)

    with pytest.raises(OperationDefinitionError) as caught:
        Operation.from_table(
            OperationSymbol("id", (value,), value),
            _carrier_bindings(carrier),
            [((stored,), _BrokenEquality())],
        )

    assert caught.value.reason == "output carrier equality comparison failed"
    assert caught.value.entry_index == 0


def test_direct_construction_is_rejected_in_favor_of_checked_factory() -> None:
    with pytest.raises(OperationDefinitionError) as caught:
        Operation(object(), (), ())

    assert caught.value.field == "operation"
    assert caught.value.reason == "use Operation.from_table for construction"


def test_operation_is_frozen_slot_backed_structural_and_deliberately_unhashable() -> (
    None
):
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    symbol = OperationSymbol("id", (bit,), bit)
    entries = [((0,), 0), ((1,), 1)]
    operation = Operation.from_table(symbol, _carrier_bindings(carrier), entries)
    equal = Operation.from_table(symbol, _carrier_bindings(carrier), reversed(entries))

    assert operation == equal
    assert not hasattr(operation, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(operation)
    with pytest.raises(FrozenInstanceError):
        operation.symbol = symbol  # type: ignore[misc]
