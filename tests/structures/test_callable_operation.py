"""Contract tests for factory-only callable total and partial operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from inspect import signature

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import (
    Operation,
    OperationApplicationError,
    OperationDefinitionError,
    PartialOperation,
    PartialOperationApplicationError,
    PartialOperationDefinitionError,
)
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.signatures import OperationSymbol


def test_callable_factories_have_explicit_public_signatures_and_total_metadata() -> (
    None
):
    """Factories make callable, carrier, totality, and bounds contracts explicit."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    calls: list[tuple[object, ...]] = []

    def flip(value: object) -> object:
        calls.append((value,))
        return 1 if value == 0 else 0

    operation = Operation.from_callable(
        OperationSymbol("flip", (bit,), bit),
        [(bit, carrier)],
        flip,
        bounds=("exact", 1),
    )

    assert tuple(signature(Operation.from_callable).parameters) == (
        "symbol",
        "carriers",
        "function",
        "bounds",
    )
    assert tuple(signature(PartialOperation.from_callable).parameters) == (
        "symbol",
        "carriers",
        "function",
        "bounds",
    )
    assert isinstance(operation, Operation)
    assert operation.domain == (carrier,)
    assert operation.totality == "total"
    assert operation.bounds == ("exact", 1)
    assert "0x" not in repr(operation)
    assert calls == []
    assert operation.apply(0) == Defined(1)
    assert calls == [(0,)]
    equal_inputs = Operation.from_callable(
        OperationSymbol("flip", (bit,), bit),
        [(bit, carrier)],
        flip,
        bounds=("exact", 1),
    )
    assert operation == operation
    assert operation != equal_inputs
    with pytest.raises(TypeError, match="unhashable"):
        hash(operation)
    with pytest.raises(FrozenInstanceError):
        operation.bounds = None  # type: ignore[misc]


def test_partial_callable_preserves_all_exact_outcome_branches_and_bounds() -> None:
    """Partial callables return exact outcome branches without totalization."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1, 2, 3), sort=bit)
    search_bounds = {"steps": 4}
    outcomes = {
        0: Defined(1),
        1: Undefined("outside declared domain", witness=(1,)),
        2: Indeterminate("search limit", bounds=search_bounds),
        3: Failed("backend fault", stage="fixture"),
    }
    operation = PartialOperation.from_callable(
        OperationSymbol("branch", (bit,), bit),
        [(bit, carrier)],
        lambda value: outcomes[value],
        bounds=search_bounds,
    )

    assert operation.totality == "partial"
    assert operation.domain == (carrier,)
    assert operation.bounds is search_bounds
    search_bounds["steps"] = 5
    assert operation.bounds == {"steps": 5}
    for value, expected in outcomes.items():
        assert operation.apply(value) is expected


def test_total_callable_preserves_computational_outcomes_but_rejects_undefined() -> (
    None
):
    """Totality forbids mathematical undefinedness, not bounded computation limits."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1, 2, 3, 4), sort=bit)
    wrapper_bounds = {"wrapper": "declared"}
    returned_bounds = {"search_steps": 3}
    defined = Defined(1)
    indeterminate = Indeterminate("search limit", bounds=returned_bounds)
    failed = Failed("backend fault", stage="fixture")
    outcomes = {
        0: 1,
        1: defined,
        2: indeterminate,
        3: failed,
        4: Undefined("mathematically outside domain"),
    }
    operation = Operation.from_callable(
        OperationSymbol("bounded_total", (bit,), bit),
        [(bit, carrier)],
        lambda value: outcomes[value],
        bounds=wrapper_bounds,
    )

    assert operation.apply(0) == Defined(1)
    assert operation.apply(1) is defined
    returned = operation.apply(2)
    assert type(returned) is Indeterminate
    assert returned is indeterminate
    assert returned.bounds is returned_bounds
    assert operation.bounds is wrapper_bounds
    assert operation.apply(3) is failed
    assert operation.apply(4) == Failed(
        "total callable returned Undefined", stage="callable"
    )


def test_callable_domains_support_varied_arities_and_unhashable_values() -> None:
    """Callable application shares validated finite-carrier boundaries with tables."""
    bit = Sort("bit")
    flag = Sort("flag")
    bits = FiniteCarrier((0, 1), sort=bit)
    flags = FiniteCarrier(("N", "Y"), sort=flag)
    extra = Sort("extra")
    extras = FiniteCarrier(("unused",), sort=extra)
    nullary = Operation.from_callable(
        OperationSymbol("one", (), bit),
        [(extra, extras), (bit, bits)],
        lambda: 1,
    )
    choose = Operation.from_callable(
        OperationSymbol("choose", (bit, flag, bit), flag),
        [(flag, flags), (bit, bits)],
        lambda _left, middle, _right: middle,
    )
    repeated = Operation.from_callable(
        OperationSymbol("left", (bit, bit), bit),
        [(bit, bits)],
        lambda left, _right: left,
    )
    unit = Sort("unit")
    units = FiniteCarrier((["only"],), sort=unit)
    high = Operation.from_callable(
        OperationSymbol("high", (unit,) * 257, unit),
        [(unit, units)],
        lambda *values: values[0],
    )

    assert nullary.carrier_bindings == ((extra, extras), (bit, bits))
    assert nullary.apply() == Defined(1)
    assert choose.apply(0, "Y", 1) == Defined("Y")
    assert repeated.apply(1, 0) == Defined(1)
    assert high.apply(*([["only"]] * 257)) == Defined(["only"])


def test_callable_factories_snapshot_generator_bindings_and_do_not_execute() -> None:
    """A carrier generator is consumed once while the callable stays lazy."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    bindings = [(bit, carrier)]
    yielded: list[tuple[Sort, FiniteCarrier]] = []
    calls = 0

    def source() -> Iterator[tuple[Sort, FiniteCarrier]]:
        for binding in bindings:
            yielded.append(binding)
            yield binding

    def flip(value: object) -> object:
        nonlocal calls
        calls += 1
        return 1 if value == 0 else 0

    operation = Operation.from_callable(
        OperationSymbol("flip", (bit,), bit), source(), flip
    )
    bindings.clear()

    assert yielded == [(bit, carrier)]
    assert operation.carrier_bindings == ((bit, carrier),)
    assert calls == 0
    assert operation.apply(0) == Defined(1)
    assert calls == 1


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (Operation.from_callable, OperationDefinitionError),
        (PartialOperation.from_callable, PartialOperationDefinitionError),
    ],
)
def test_callable_factory_rejects_malformed_symbol_carriers_and_callable(
    factory: object, error_type: type[OperationDefinitionError]
) -> None:
    """Definition failures remain typed instead of becoming evaluation outcomes."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    callable_factory = factory  # narrowed below for concise parametrization

    with pytest.raises(error_type) as bad_symbol:
        callable_factory(object(), [(bit, carrier)], lambda _value: 0)  # type: ignore[operator]
    assert bad_symbol.value.field == "symbol"
    with pytest.raises(error_type) as bad_carrier:
        callable_factory(OperationSymbol("id", (bit,), bit), [], lambda _value: 0)  # type: ignore[operator]
    assert bad_carrier.value.field == "carriers"
    with pytest.raises(error_type) as bad_callable:
        callable_factory(OperationSymbol("id", (bit,), bit), [(bit, carrier)], object())  # type: ignore[operator]
    assert bad_callable.value.field == "function"


def test_callable_apply_rejects_invalid_arity_and_carrier_members_before_calling() -> (
    None
):
    """Strict application validation is unchanged for callable-backed operations."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    calls: list[object] = []

    def record(value: object) -> object:
        calls.append(value)
        return value

    total = Operation.from_callable(
        OperationSymbol("id", (bit,), bit),
        [(bit, carrier)],
        record,
    )
    partial = PartialOperation.from_callable(
        OperationSymbol("id", (bit,), bit),
        [(bit, carrier)],
        lambda value: Defined(value),
    )

    with pytest.raises(OperationApplicationError):
        total.apply()
    with pytest.raises(OperationApplicationError):
        total.apply(3)
    with pytest.raises(PartialOperationApplicationError):
        partial.apply()
    with pytest.raises(PartialOperationApplicationError):
        partial.apply(3)
    assert calls == []


def test_callable_result_validation_and_exception_translation_are_safe() -> None:
    """Bad results and exceptions produce inert Failed records, never Undefined."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    total_bad_output = Operation.from_callable(
        OperationSymbol("bad", (bit,), bit), [(bit, carrier)], lambda _value: 2
    )
    total_raises = Operation.from_callable(
        OperationSymbol("raise", (bit,), bit),
        [(bit, carrier)],
        lambda _value: (_ for _ in ()).throw(ValueError("secret detail")),
    )
    partial_bad_branch = PartialOperation.from_callable(
        OperationSymbol("bad", (bit,), bit), [(bit, carrier)], lambda _value: 1
    )
    partial_bad_defined = PartialOperation.from_callable(
        OperationSymbol("bad-defined", (bit,), bit),
        [(bit, carrier)],
        lambda _value: Defined(2),
    )
    partial_raises = PartialOperation.from_callable(
        OperationSymbol("raise", (bit,), bit),
        [(bit, carrier)],
        lambda _value: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )

    for outcome in (
        total_bad_output.apply(0),
        partial_bad_branch.apply(0),
        partial_bad_defined.apply(0),
    ):
        assert type(outcome) is Failed
        assert outcome.stage == "callable"
        assert "secret detail" not in outcome.error
    total_failure = total_raises.apply(0)
    partial_failure = partial_raises.apply(0)
    assert total_failure == Failed("callable raised ValueError", stage="callable")
    assert partial_failure == Failed("callable raised RuntimeError", stage="callable")


def test_callable_exception_translation_sanitizes_malformed_type_names() -> None:
    """A mutable invalid exception class name cannot escape the Failed boundary."""
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)

    class FixtureError(Exception):
        """A test exception whose runtime type label is deliberately malformed."""

    FixtureError.__name__ = " unsafe name "
    operation = Operation.from_callable(
        OperationSymbol("raise", (bit,), bit),
        [(bit, carrier)],
        lambda _value: (_ for _ in ()).throw(FixtureError()),
    )

    assert operation.apply(0) == Failed("callable raised exception", stage="callable")
