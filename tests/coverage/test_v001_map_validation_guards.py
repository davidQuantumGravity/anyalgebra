"""Close the observable and invariant branches of finite map validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

import anyalgebra.maps.validate as validation
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.maps.core import Map, Morphism
from anyalgebra.maps.validate import (
    Counterexample,
    Disproved,
    Inconclusive,
    Proved,
    ValidationCounts,
    ValidationScope,
    validate_homomorphism,
    validate_inverse,
    validate_isomorphism,
)
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.relations import Relation, RelationResult
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder


def _carrier_structure(items: tuple[object, ...]) -> Structure:
    sort = Sort("value")
    return (
        StructureBuilder(Signature((sort,)))
        .with_carrier(sort, FiniteCarrier(items, sort=sort))
        .freeze()
    )


def _morphism(
    source: Structure, target: Structure, function: Callable[[object], object]
) -> Morphism:
    return Morphism.from_components(
        source,
        target,
        (Map.from_callable(source.carriers[0], target.carriers[0], function),),
    )


def _operation_structure(
    function: Callable[[object], object],
    *,
    symbol: OperationSymbol | None = None,
) -> tuple[Structure, OperationSymbol]:
    sort = Sort("value")
    operation_symbol = symbol or OperationSymbol("op", (sort,), sort)
    carrier = FiniteCarrier((0, 1), sort=sort)
    operation = Operation.from_callable(operation_symbol, ((sort, carrier),), function)
    structure = (
        StructureBuilder(Signature((sort,), operations=(operation_symbol,)))
        .with_carrier(sort, carrier)
        .with_operation(operation_symbol, operation)
        .freeze()
    )
    return structure, operation_symbol


def _relation_structure(
    predicate: Callable[[object], object],
    *,
    symbol: RelationSymbol | None = None,
) -> tuple[Structure, RelationSymbol]:
    sort = Sort("value")
    relation_symbol = symbol or RelationSymbol("rel", (sort,))
    carrier = FiniteCarrier((0, 1), sort=sort)
    relation = Relation.from_predicate(relation_symbol, ((sort, carrier),), predicate)
    structure = (
        StructureBuilder(Signature((sort,), relations=(relation_symbol,)))
        .with_carrier(sort, carrier)
        .with_relation(relation_symbol, relation)
        .freeze()
    )
    return structure, relation_symbol


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_symbols", []),
        ("relation_symbols", (object(),)),
        ("inverse_sorts", (object(),)),
        ("source_carrier_sizes", (True,)),
        ("target_carrier_sizes", (-1,)),
    ],
)
def test_scope_rejects_every_malformed_tuple_field(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "operation_symbols": (),
        "relation_symbols": (),
        field: value,
    }
    with pytest.raises((TypeError, ValueError)):
        ValidationScope(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["description", "algorithm", "enumeration_order"])
@pytest.mark.parametrize("value", ["", " padded ", cast(str, 1)])
def test_scope_rejects_malformed_text(field: str, value: str) -> None:
    arguments: dict[str, object] = {
        "operation_symbols": (),
        "relation_symbols": (),
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        ValidationScope(**arguments)  # type: ignore[arg-type]


def test_scope_rejects_nonstructure_endpoints() -> None:
    with pytest.raises(TypeError, match="source"):
        ValidationScope((), (), source=cast(Structure, object()))
    with pytest.raises(TypeError, match="target"):
        ValidationScope((), (), target=cast(Structure, object()))


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_counts_reject_non_counts(value: object) -> None:
    with pytest.raises(ValueError, match="operation_expected"):
        ValidationCounts(cast(int, value), 0, 0, 0)


@pytest.mark.parametrize("position", range(4))
def test_counts_reject_each_evaluated_over_expected_pair(position: int) -> None:
    values = [0] * 8
    values[position * 2 + 1] = 1
    with pytest.raises(ValueError, match="exceeds"):
        ValidationCounts(*values)


def _zero_counts() -> ValidationCounts:
    return ValidationCounts(0, 0, 0, 0)


def _zero_scope() -> ValidationScope:
    return ValidationScope((), ())


def test_report_record_type_and_exhaustiveness_guards_and_properties() -> None:
    with pytest.raises(TypeError, match="exact scope"):
        Proved(cast(ValidationScope, object()), _zero_counts())
    with pytest.raises(TypeError, match="exact scope"):
        Proved(_zero_scope(), cast(ValidationCounts, object()))
    for counts in (
        ValidationCounts(1, 0, 0, 0),
        ValidationCounts(0, 0, 1, 0),
        ValidationCounts(0, 0, 0, 0, 1, 0),
        ValidationCounts(0, 0, 0, 0, 0, 0, 1, 0),
    ):
        with pytest.raises(ValueError, match="exhaustive"):
            Proved(_zero_scope(), counts)

    witness = Counterexample("kind", None, None, ())
    report_counts = ValidationCounts(1, 1, 2, 2, 3, 3, 4, 4)
    disproved = Disproved(witness, _zero_scope(), report_counts)
    inconclusive = Inconclusive("unknown", _zero_scope(), report_counts)
    for report in (disproved, inconclusive):
        assert report.operation_cases == 1
        assert report.relation_cases == 2
        assert report.inverse_cases == 3
        assert report.mapping_cases == 4
    proved = Proved(_zero_scope(), report_counts)
    assert proved.mapping_cases == 4


@pytest.mark.parametrize("kind", ["", " padded ", cast(str, 1)])
def test_counterexample_rejects_bad_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        Counterexample(kind, None, None, ())


def test_counterexample_rejects_bad_symbol_sort_and_inputs() -> None:
    with pytest.raises(TypeError, match="symbol"):
        Counterexample("kind", cast(OperationSymbol, object()), None, ())
    with pytest.raises(TypeError, match="sort"):
        Counterexample("kind", None, cast(Sort, object()), ())
    with pytest.raises(TypeError, match="inputs"):
        Counterexample("kind", None, None, cast(tuple[object, ...], []))


@pytest.mark.parametrize("position", range(3))
def test_disproved_rejects_each_bad_record(position: int) -> None:
    values: list[object] = [
        Counterexample("kind", None, None, ()),
        _zero_scope(),
        _zero_counts(),
    ]
    values[position] = object()
    with pytest.raises(TypeError, match="exact witness"):
        Disproved(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["", " padded ", cast(str, 1)])
def test_inconclusive_rejects_bad_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        Inconclusive(reason, _zero_scope(), _zero_counts())


@pytest.mark.parametrize("position", [1, 2])
def test_inconclusive_rejects_bad_records(position: int) -> None:
    values: list[object] = ["unknown", _zero_scope(), _zero_counts()]
    values[position] = object()
    with pytest.raises(TypeError, match="exact scope"):
        Inconclusive(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (Defined(0), None),
        (Undefined("u"), "Undefined"),
        (Indeterminate("i"), "Indeterminate"),
        (Failed("f"), "Failed"),
        (2, "outside"),
    ],
)
def test_component_result_normalization_is_visible(
    result: object, reason: str | None
) -> None:
    source = _carrier_structure((0,))
    target = _carrier_structure((0,))
    report = validate_homomorphism(_morphism(source, target, lambda _x: result))
    if reason == "outside":
        assert type(report) is Disproved
        assert report.counterexample.kind == "invalid_component_image"
    elif reason is None:
        assert type(report) is Proved
    else:
        assert type(report) is Inconclusive
        assert reason in report.reason


def test_invalid_exception_name_is_sanitized() -> None:
    bad_exception = type("bad-name", (Exception,), {})
    source = _carrier_structure((0,))
    target = _carrier_structure((0,))

    def fail(_value: object) -> object:
        raise bad_exception("private")

    report = validate_homomorphism(_morphism(source, target, fail))
    assert type(report) is Inconclusive
    assert report.reason.endswith("exception")


def test_operation_source_target_and_mapping_unknown_paths() -> None:
    symbol = OperationSymbol("op", (Sort("value"),), Sort("value"))
    source_unknown, _ = _operation_structure(lambda _x: Undefined("u"), symbol=symbol)
    target_defined, _ = _operation_structure(lambda x: x, symbol=symbol)
    report = validate_homomorphism(
        _morphism(source_unknown, target_defined, lambda x: x)
    )
    assert type(report) is Inconclusive
    assert "source operation" in report.reason

    source_defined, _ = _operation_structure(lambda _x: 0, symbol=symbol)
    target_unknown, _ = _operation_structure(
        lambda _x: Indeterminate("i"), symbol=symbol
    )
    report = validate_homomorphism(
        _morphism(source_defined, target_unknown, lambda x: x)
    )
    assert type(report) is Inconclusive
    assert "target operation" in report.reason

    report = validate_homomorphism(
        _morphism(
            source_defined,
            target_defined,
            lambda x: 0 if x == 0 else Failed("unknown"),
        )
    )
    assert type(report) is Inconclusive
    assert "component returned Failed" in report.reason


def test_relation_source_target_unknown_and_counterexample_paths() -> None:
    symbol = RelationSymbol("rel", (Sort("value"),))
    source_unknown, _ = _relation_structure(
        lambda _x: Indeterminate("i"), symbol=symbol
    )
    target_true, _ = _relation_structure(lambda _x: True, symbol=symbol)
    report = validate_homomorphism(_morphism(source_unknown, target_true, lambda x: x))
    assert type(report) is Inconclusive
    assert "source relation" in report.reason

    source_true, _ = _relation_structure(lambda _x: True, symbol=symbol)
    target_unknown, _ = _relation_structure(lambda _x: Failed("f"), symbol=symbol)
    report = validate_homomorphism(_morphism(source_true, target_unknown, lambda x: x))
    assert type(report) is Inconclusive
    assert "target relation" in report.reason

    target_false, _ = _relation_structure(lambda _x: False, symbol=symbol)
    report = validate_homomorphism(_morphism(source_true, target_false, lambda x: x))
    assert type(report) is Disproved
    assert report.counterexample.kind == "relation"

    report = validate_homomorphism(
        _morphism(source_true, target_true, lambda x: 0 if x == 0 else Failed("f"))
    )
    assert type(report) is Inconclusive


def test_internal_application_guards_normalize_exceptions_and_unknown_outcomes() -> (
    None
):
    class Raises:
        def apply(self, *_inputs: object) -> object:
            raise RuntimeError("private")

    class Unknown:
        def apply(self, *_inputs: object) -> object:
            return object()

    assert validation._apply_operation(cast(Operation, Raises()), (), "op") == (
        False,
        "op application raised RuntimeError",
    )
    assert validation._apply_operation(cast(Operation, Unknown()), (), "op") == (
        False,
        "op returned an unsupported outcome",
    )
    assert validation._apply_relation(cast(Relation, Raises()), (), "rel") == (
        False,
        "rel application raised RuntimeError",
    )
    assert validation._apply_relation(cast(Relation, Unknown()), (), "rel") == (
        False,
        "rel returned an unsupported outcome",
    )
    assert validation._apply_relation(
        cast(
            Relation, type("Result", (), {"apply": lambda self: RelationResult(True)})()
        ),
        (),
        "rel",
    ) == (True, True)


def test_internal_lookup_invariants_fail_loudly() -> None:
    source = _carrier_structure((0,))
    morphism = _morphism(source, source, lambda x: x)
    with pytest.raises(AssertionError, match="declared sort"):
        validation._sort_index(morphism, Sort("missing"))
    operation_symbol = OperationSymbol("missing", (), source.signature.sorts[0])
    with pytest.raises(AssertionError, match="declared operation"):
        validation._operation_for((), operation_symbol)
    relation_symbol = RelationSymbol("missing", ())
    with pytest.raises(AssertionError, match="declared relation"):
        validation._relation_for((), relation_symbol)


def test_internal_cache_lookup_and_explicit_count_paths() -> None:
    source = _carrier_structure((0,))
    morphism = _morphism(source, source, lambda x: x)
    cache = validation._ImageCache(morphism)
    assert cache.evaluated == 0
    assert cache.image(0, 0).state == "defined"
    assert cache.image(0, 0).state == "defined"
    assert cache.evaluated == 1
    counts = validation._counts(morphism, 0, 0, mapping_expected=1)
    assert counts.mapping_expected == counts.mapping_evaluated == 1


def test_internal_lookup_scans_past_an_earlier_symbol() -> None:
    source, wanted_operation = _operation_structure(lambda x: x)
    other_source, _ = _operation_structure(
        lambda x: x,
        symbol=OperationSymbol("other", (Sort("value"),), Sort("value")),
    )
    assert (
        validation._operation_for(
            (other_source.operations[0], source.operations[0]), wanted_operation
        )
        is source.operations[0]
    )

    relation_source, wanted_relation = _relation_structure(lambda _x: True)
    other_relation_source, _ = _relation_structure(
        lambda _x: True,
        symbol=RelationSymbol("other", (Sort("value"),)),
    )
    assert (
        validation._relation_for(
            (other_relation_source.relations[0], relation_source.relations[0]),
            wanted_relation,
        )
        is relation_source.relations[0]
    )


def test_public_validators_reject_non_morphisms() -> None:
    bad = cast(Morphism, object())
    assert type(validate_homomorphism(bad)) is Inconclusive
    assert type(validate_inverse(bad, bad)) is Inconclusive
    assert type(validate_isomorphism(bad, bad)) is Inconclusive


def test_inverse_invalid_and_unknown_images_in_both_directions() -> None:
    source = _carrier_structure((0, 1))
    target = _carrier_structure((0, 1, 2))

    forward_invalid = _morphism(source, target, lambda _x: 3)
    backward = _morphism(target, source, lambda x: cast(int, x) % 2)
    report = validate_inverse(forward_invalid, backward)
    assert type(report) is Disproved

    forward_unknown = _morphism(source, target, lambda _x: Failed("f"))
    report = validate_inverse(forward_unknown, backward)
    assert type(report) is Inconclusive

    forward = _morphism(source, target, lambda x: x)
    backward_invalid = _morphism(
        target, source, lambda x: 2 if x == 0 else cast(int, x) % 2
    )
    report = validate_inverse(forward, backward_invalid)
    assert type(report) is Disproved

    backward_unknown = _morphism(
        target,
        source,
        lambda x: x if x == 1 else Indeterminate("i"),
    )
    report = validate_inverse(forward, backward_unknown)
    assert type(report) is Inconclusive

    extra_invalid = _morphism(target, source, lambda x: x if cast(int, x) < 2 else 2)
    report = validate_inverse(forward, extra_invalid)
    assert type(report) is Disproved
    assert report.counterexample.kind == "invalid_component_image"

    extra_unknown = _morphism(
        target,
        source,
        lambda x: x if cast(int, x) < 2 else Undefined("u"),
    )
    report = validate_inverse(forward, extra_unknown)
    assert type(report) is Inconclusive

    all_backward_unknown = _morphism(target, source, lambda _x: Failed("f"))
    report = validate_inverse(forward, all_backward_unknown)
    assert type(report) is Inconclusive


def test_inverse_rejects_nonreversed_endpoints() -> None:
    source = _carrier_structure((0,))
    target = _carrier_structure((0,))
    forward = _morphism(source, target, lambda x: x)
    same_direction = _morphism(source, target, lambda x: x)
    report = validate_inverse(forward, same_direction)
    assert type(report) is Inconclusive
    assert "reversed" in report.reason


def test_inverse_left_and_right_identity_witnesses() -> None:
    carrier = _carrier_structure((0, 1))
    identity = _morphism(carrier, carrier, lambda x: x)
    constant = _morphism(carrier, carrier, lambda _x: 0)
    left = validate_inverse(identity, constant)
    assert type(left) is Disproved
    assert left.counterexample.kind == "left_inverse"

    source = _carrier_structure((0, 1))
    target = _carrier_structure((0, 1, 2))
    inclusion = _morphism(source, target, lambda x: x)
    retraction = _morphism(target, source, lambda x: x if cast(int, x) < 2 else 0)
    right = validate_inverse(inclusion, retraction)
    assert type(right) is Disproved
    assert right.counterexample.kind == "right_inverse"


def test_isomorphism_returns_first_disproof_then_first_inconclusive() -> None:
    source, symbol = _operation_structure(lambda x: x)
    target, _ = _operation_structure(lambda _x: 0, symbol=symbol)
    forward = _morphism(source, target, lambda x: x)
    backward = _morphism(target, source, lambda x: x)
    report = validate_isomorphism(forward, backward)
    assert type(report) is Disproved

    plain = _carrier_structure((0, 1))
    unknown = _morphism(plain, plain, lambda _x: Failed("f"))
    identity = _morphism(plain, plain, lambda x: x)
    report = validate_isomorphism(unknown, identity)
    assert type(report) is Inconclusive
