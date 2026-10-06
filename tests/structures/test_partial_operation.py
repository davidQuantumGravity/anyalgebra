"""Contract tests for finite, table-backed partial operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import (
    Operation,
    OperationDefinitionError,
    PartialOperation,
    PartialOperationApplicationError,
    PartialOperationDefinitionError,
    TableEntries,
)
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.signatures import OperationSymbol


def test_partial_table_distinguishes_explicit_and_missing_cells() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    marker = object()
    operation = PartialOperation.from_table(
        OperationSymbol("partial_flip", (bit,), bit),
        [(bit, carrier)],
        [((0,), 1), ((1,), marker)],
        undefined_marker=marker,
    )

    assert operation.apply(0) == Defined(1)
    assert operation.apply(1) == Undefined(
        "table cell explicitly marked undefined", witness=(1,)
    )


def test_partial_table_missing_cartesian_cells_are_undefined_without_bottom() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    marker = object()
    operation = PartialOperation.from_table(
        OperationSymbol("partial_id", (bit,), bit),
        [(bit, carrier)],
        [((0,), 0)],
        undefined_marker=marker,
    )

    assert operation.output_carrier is carrier
    assert operation.output_carrier.items == (0, 1)
    assert operation.apply(1) == Undefined("table cell is not declared", witness=(1,))


def test_dense_all_defined_partial_table_has_no_undefined_cells() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = PartialOperation.from_table(
        OperationSymbol("flip", (bit,), bit),
        [(bit, carrier)],
        [((0,), 1), ((1,), 0)],
        undefined_marker=object(),
    )

    assert operation.undefined_indices == ()
    assert operation.apply(0) == Defined(1)
    assert operation.apply(1) == Defined(0)


def test_nullary_repeated_sort_and_many_sorted_partial_tables_apply_exactly() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    marker = object()

    missing_constant = PartialOperation.from_table(
        OperationSymbol("missing", (), bit), [(bit, bits)], [], undefined_marker=marker
    )
    marked_constant = PartialOperation.from_table(
        OperationSymbol("marked", (), bit),
        [(bit, bits)],
        [((), marker)],
        undefined_marker=marker,
    )
    choose = PartialOperation.from_table(
        OperationSymbol("choose", (bit, flag, bit), flag),
        [(bit, bits), (flag, flags)],
        [((0, "Y", 1), "Y")],
        undefined_marker=marker,
    )
    repeated = PartialOperation.from_table(
        OperationSymbol("left", (bit, bit), bit),
        [(bit, bits)],
        [((0, 1), 0)],
        undefined_marker=marker,
    )
    unit = Sort("unit")
    units = FiniteCarrier(("only",), sort=unit)
    high = PartialOperation.from_table(
        OperationSymbol("high", (unit,) * 257, unit),
        [(unit, units)],
        [(tuple("only" for _ in range(257)), "only")],
        undefined_marker=marker,
    )

    assert missing_constant.apply() == Undefined("table cell is not declared", ())
    assert marked_constant.apply() == Undefined(
        "table cell explicitly marked undefined", ()
    )
    assert choose.apply(0, "Y", 1) == Defined("Y")
    assert choose.apply(0, "N", 1) == Undefined("table cell is not declared", (0, 0, 1))
    assert repeated.apply(0, 1) == Defined(0)
    assert high.apply(*(["only"] * 257)) == Defined("only")


def test_unhashable_carriers_and_identity_only_marker_collision_semantics() -> None:
    list_sort = Sort("list")
    declared_output: list[int] = [0]
    marker: list[int] = [0]
    carrier = FiniteCarrier((declared_output, [1]), sort=list_sort)
    operation = PartialOperation.from_table(
        OperationSymbol("select", (list_sort,), list_sort),
        [(list_sort, carrier)],
        [(([0],), declared_output), (([1],), marker)],
        undefined_marker=marker,
    )

    assert operation.apply([0]) == Defined(declared_output)
    assert operation.apply([1]) == Undefined(
        "table cell explicitly marked undefined", (1,)
    )


def test_marker_identical_to_an_output_carrier_item_is_rejected() -> None:
    bit = Sort("bit")
    marker = object()
    carrier = FiniteCarrier((marker,), sort=bit)

    with pytest.raises(PartialOperationDefinitionError) as caught:
        PartialOperation.from_table(
            OperationSymbol("constant", (), bit),
            [(bit, carrier)],
            [],
            undefined_marker=marker,
        )

    assert caught.value.field == "undefined_marker"
    assert (
        caught.value.reason == "must not be identical to a declared output carrier item"
    )
    assert caught.value.index_tuple == (0,)
    assert "0x" not in str(caught.value)


def test_sources_are_snapshotted_once_and_extra_carriers_are_retained() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    bindings = [(flag, flags), (bit, bits)]
    entries = [((0,), 1)]
    yielded_bindings: list[tuple[Sort, FiniteCarrier]] = []
    yielded_entries: list[tuple[tuple[int], int]] = []

    def carrier_bindings() -> Iterator[tuple[Sort, FiniteCarrier]]:
        for binding in bindings:
            yielded_bindings.append(binding)
            yield binding

    def table() -> Iterator[tuple[tuple[int], int]]:
        for entry in entries:
            yielded_entries.append(entry)
            yield entry

    operation = PartialOperation.from_table(
        OperationSymbol("flip_once", (bit,), bit),
        carrier_bindings(),
        table(),
        undefined_marker=object(),
    )
    bindings.clear()
    entries[:] = [((0,), 0)]

    assert yielded_bindings == [(flag, flags), (bit, bits)]
    assert yielded_entries == [((0,), 1)]
    assert operation.carrier_bindings == ((flag, flags), (bit, bits))
    assert operation.apply(0) == Defined(1)


def test_mapping_sources_are_accepted_and_canonicalized() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = PartialOperation.from_table(
        OperationSymbol("flip", (bit,), bit),
        {bit: carrier},
        cast(TableEntries, {(1,): 0, (0,): 1}),
        undefined_marker=object(),
    )

    assert operation.table == (((0,), 1), ((1,), 0))
    assert operation.apply(0) == Defined(1)


@pytest.mark.parametrize(
    ("bindings", "reason", "binding_index"),
    [
        ("missing", "missing required carrier binding", None),
        ("duplicate", "duplicate carrier binding", 1),
        ("mismatch", "carrier sort does not match binding sort", 0),
    ],
)
def test_partial_carrier_binding_failures_are_typed(
    bindings: str, reason: str, binding_index: int | None
) -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    declarations = {
        "missing": [(bit, bits)],
        "duplicate": [(bit, bits), (bit, bits), (flag, flags)],
        "mismatch": [(bit, flags), (flag, flags)],
    }

    with pytest.raises(PartialOperationDefinitionError) as caught:
        PartialOperation.from_table(
            OperationSymbol("choose", (bit, flag), bit),
            declarations[bindings],
            [],
            undefined_marker=object(),
        )

    assert caught.value.reason == reason
    assert caught.value.binding_index == binding_index


@pytest.mark.parametrize(
    ("table", "reason", "entry_index"),
    [
        ([(0,)], "entry must be a pair", 0),
        ([([0], 0)], "input key must be an exact tuple", 0),
        ([((0, 1), 0)], "input tuple has wrong arity", 0),
        ([((2,), 0)], "input is not a member of its declared carrier", 0),
        ([((0,), 2)], "output is not a member of its declared carrier", 0),
    ],
)
def test_malformed_partial_tables_raise_typed_address_free_errors(
    table: object, reason: str, entry_index: int
) -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)

    with pytest.raises(PartialOperationDefinitionError) as caught:
        PartialOperation.from_table(
            OperationSymbol("partial_id", (bit,), bit),
            [(bit, carrier)],
            cast(TableEntries, table),
            undefined_marker=object(),
        )

    assert isinstance(caught.value, AnyAlgebraError)
    assert (caught.value.reason, caught.value.entry_index) == (reason, entry_index)
    assert "0x" not in str(caught.value)


def test_duplicate_cells_and_invalid_requests_are_typed_before_lookup() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    marker = object()
    with pytest.raises(PartialOperationDefinitionError) as duplicate:
        PartialOperation.from_table(
            OperationSymbol("id", (bit,), bit),
            [(bit, carrier)],
            [((0,), marker), ((0,), 0)],
            undefined_marker=marker,
        )
    assert duplicate.value.reason == "duplicate table input tuple"
    assert duplicate.value.entry_index == 1
    assert duplicate.value.conflicting_entry_index == 0
    assert duplicate.value.index_tuple == (0,)

    operation = PartialOperation.from_table(
        OperationSymbol("id", (bit,), bit),
        [(bit, carrier)],
        [((0,), 0)],
        undefined_marker=marker,
    )
    with pytest.raises(PartialOperationApplicationError) as wrong_arity:
        operation.apply()
    assert (wrong_arity.value.expected_arity, wrong_arity.value.actual_arity) == (1, 0)
    with pytest.raises(PartialOperationApplicationError) as wrong_member:
        operation.apply(3)
    assert (
        wrong_member.value.reason == "argument is not a member of its declared carrier"
    )
    assert wrong_member.value.argument_index == 0
    assert wrong_member.value.expected_sort == bit


class _BrokenEquality:
    """A value that makes non-identical finite-carrier equality unavailable."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_partial_carrier_equality_failures_remain_typed() -> None:
    value = Sort("value")
    stored = _BrokenEquality()
    carrier = FiniteCarrier((stored,), sort=value)

    with pytest.raises(PartialOperationDefinitionError) as output_failure:
        PartialOperation.from_table(
            OperationSymbol("id", (value,), value),
            [(value, carrier)],
            [((stored,), _BrokenEquality())],
            undefined_marker=object(),
        )
    assert output_failure.value.reason == "output carrier equality comparison failed"

    operation = PartialOperation.from_table(
        OperationSymbol("id", (value,), value),
        [(value, carrier)],
        [((stored,), stored)],
        undefined_marker=object(),
    )
    with pytest.raises(PartialOperationApplicationError) as application_failure:
        operation.apply(_BrokenEquality())
    assert (
        application_failure.value.reason
        == "argument carrier equality comparison failed"
    )


def test_complete_table_operation_still_rejects_a_missing_cell() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)

    with pytest.raises(OperationDefinitionError) as caught:
        Operation.from_table(
            OperationSymbol("total_id", (bit,), bit), [(bit, carrier)], [((0,), 0)]
        )

    assert caught.value.reason == "table is missing required input tuple"
    assert caught.value.index_tuple == (1,)


def test_partial_operation_is_factory_only_frozen_and_unhashable() -> None:
    with pytest.raises(PartialOperationDefinitionError) as caught:
        PartialOperation(object(), (), (), ())
    assert caught.value.reason == "use PartialOperation.from_table for construction"

    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    symbol = OperationSymbol("id", (bit,), bit)
    operation = PartialOperation.from_table(
        symbol, [(bit, carrier)], [((0,), 0)], undefined_marker=object()
    )
    equal = PartialOperation.from_table(
        symbol, [(bit, carrier)], [((0,), 0)], undefined_marker=object()
    )
    assert operation == equal
    assert not hasattr(operation, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(operation)
    with pytest.raises(FrozenInstanceError):
        operation.symbol = symbol  # type: ignore[misc]
