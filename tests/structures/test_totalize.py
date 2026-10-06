"""Contract tests for explicit strict-bottom totalization of finite tables."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.partiality import (
    Totalization,
    TotalizationDefinitionError,
    totalize,
)
from anyalgebra.structures.signatures import OperationSymbol


def _defined_value(operation: Operation, *arguments: object) -> object:
    outcome = operation.apply(*arguments)
    assert type(outcome) is Defined
    return outcome.value


def test_totalize_maps_explicit_and_omitted_undefined_cells_to_bottom() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    marker = object()
    source = PartialOperation.from_table(
        OperationSymbol("partial", (bit,), bit),
        [(bit, carrier)],
        [((0,), 0), ((1,), marker)],
        undefined_marker=marker,
    )
    result = totalize(source, bottom=2)

    assert result.source is source
    assert result.bottom == 2
    assert result.strict_policy == "strict-bottom"
    assert result.totalized_carrier.items == (0, 1, 2)
    assert _defined_value(result.operation, 0) == 0
    assert _defined_value(result.operation, 1) == 2
    assert _defined_value(result.operation, 2) == 2
    assert source.apply(1) == Undefined(
        "table cell explicitly marked undefined", witness=(1,)
    )


def test_totalize_nullary_output_absent_binary_and_high_arity_tables() -> None:
    output = Sort("output")
    input_sort = Sort("input")
    outputs = FiniteCarrier(("ok",), sort=output)
    inputs = FiniteCarrier((0, 1), sort=input_sort)
    nullary = PartialOperation.from_table(
        OperationSymbol("constant", (), output),
        [(output, outputs)],
        [],
        undefined_marker=object(),
    )
    binary = PartialOperation.from_table(
        OperationSymbol("choose", (input_sort, input_sort), output),
        [(input_sort, inputs), (output, outputs)],
        [((0, 0), "ok")],
        undefined_marker=object(),
    )
    unit = Sort("unit")
    high_output = Sort("high_output")
    units = FiniteCarrier((0,), sort=unit)
    high_outputs = FiniteCarrier((0,), sort=high_output)
    high = PartialOperation.from_table(
        OperationSymbol("high", (unit,) * 257, high_output),
        [(unit, units), (high_output, high_outputs)],
        [(tuple(0 for _ in range(257)), 0)],
        undefined_marker=object(),
    )

    assert _defined_value(totalize(nullary, bottom="bottom").operation) == "bottom"
    chosen = totalize(binary, bottom="bottom")
    assert _defined_value(chosen.operation, 0, 0) == "ok"
    assert _defined_value(chosen.operation, 1, 0) == "bottom"
    assert _defined_value(totalize(high, bottom=1).operation, *([0] * 257)) == 0


def test_totalize_replaces_output_sort_everywhere_and_is_cartesian_complete() -> None:
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    source = PartialOperation.from_table(
        OperationSymbol("mix", (bit, flag, bit), bit),
        [(flag, flags), (bit, bits)],
        [((0, "Y", 1), 1)],
        undefined_marker=object(),
    )
    result = totalize(source, bottom=2)

    assert result.operation.carrier_bindings == (
        (flag, flags),
        (bit, result.totalized_carrier),
    )
    assert result.operation.input_carriers == (
        result.totalized_carrier,
        flags,
        result.totalized_carrier,
    )
    assert result.operation.output_carrier is result.totalized_carrier
    assert len(result.operation.table) == 3 * 2 * 3
    assert _defined_value(result.operation, 2, "Y", 1) == 2
    assert _defined_value(result.operation, 0, "N", 1) == 2


def test_totalize_supports_unhashable_bottom_extra_bindings_and_embedding() -> None:
    value = Sort("value")
    extra = Sort("extra")
    values = FiniteCarrier(([0], [1]), sort=value)
    extras = FiniteCarrier(("unused",), sort=extra)
    source = PartialOperation.from_table(
        OperationSymbol("id", (value,), value),
        [(extra, extras), (value, values)],
        [(([0],), [0])],
        undefined_marker=object(),
    )
    bottom = ["bottom"]
    result = totalize(source, bottom=bottom)

    assert result.operation.carrier_bindings[0] == (extra, extras)
    assert _defined_value(result.operation, [1]) is bottom
    assert result.embedding.embed([0]) == [0]
    with pytest.raises(TotalizationDefinitionError) as missing:
        result.embedding.embed([99])
    assert missing.value.reason == "embedding value is not a source output member"


class _BrokenEquality:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison unavailable")


def test_totalize_rejects_collisions_equality_failure_callable_and_wrong_types() -> (
    None
):
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    source = PartialOperation.from_table(
        OperationSymbol("id", (bit,), bit),
        [(bit, carrier)],
        [],
        undefined_marker=object(),
    )
    with pytest.raises(TotalizationDefinitionError) as collision:
        totalize(source, bottom=0)
    assert collision.value.reason == "bottom is already an output carrier member"
    with pytest.raises(TotalizationDefinitionError) as wrong:
        totalize(cast(PartialOperation, object()), bottom=2)
    assert wrong.value.reason == "partial operation must be an exact PartialOperation"
    callable_partial = PartialOperation.from_callable(
        source.symbol, [(bit, carrier)], lambda _value: 0
    )
    with pytest.raises(TotalizationDefinitionError) as callable_rejected:
        totalize(callable_partial, bottom=2)
    assert (
        callable_rejected.value.reason
        == "callable partial operations cannot be totalized"
    )

    value = Sort("value")
    stored = _BrokenEquality()
    broken = PartialOperation.from_table(
        OperationSymbol("broken", (value,), value),
        [(value, FiniteCarrier((stored,), sort=value))],
        [((stored,), stored)],
        undefined_marker=object(),
    )
    with pytest.raises(TotalizationDefinitionError) as equality:
        totalize(broken, bottom=_BrokenEquality())
    assert equality.value.reason == "bottom carrier equality comparison failed"


def test_totalization_records_are_factory_only_frozen_structural_and_unhashable() -> (
    None
):
    with pytest.raises(TotalizationDefinitionError):
        Totalization(
            object(), object(), object(), object(), object(), object(), object()
        )
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    source = PartialOperation.from_table(
        OperationSymbol("id", (bit,), bit),
        [(bit, carrier)],
        [],
        undefined_marker=object(),
    )
    result = totalize(source, bottom=2)
    equal = totalize(source, bottom=2)

    assert result == equal
    assert not hasattr(result, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    with pytest.raises(FrozenInstanceError):
        result.bottom = 3  # type: ignore[misc]
