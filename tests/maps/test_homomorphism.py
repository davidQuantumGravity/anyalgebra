"""Finite exact preservation and inverse checks for component morphisms."""

from __future__ import annotations

from typing import cast

import pytest

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
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined
from anyalgebra.structures.relations import Relation
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder


def _structure(*, broken_target: bool = False) -> Structure:
    sort = Sort("value")
    operation_symbol = OperationSymbol("flip", (sort,), sort)
    relation_symbol = RelationSymbol("marked", (sort,))
    signature = Signature(
        (sort,), operations=(operation_symbol,), relations=(relation_symbol,)
    )
    carrier = FiniteCarrier((0, 1), sort=sort)
    operation = Operation.from_table(
        operation_symbol,
        ((sort, carrier),),
        (((0,), 1 if not broken_target else 0), ((1,), 0)),
    )
    relation = Relation.from_tuples(relation_symbol, ((sort, carrier),), ((0,),))
    return (
        StructureBuilder(signature)
        .with_carrier(sort, carrier)
        .with_operation(operation_symbol, operation)
        .with_relation(relation_symbol, relation)
        .freeze()
    )


def test_exact_operation_relation_preservation_and_inverse() -> None:
    source = _structure()
    target = _structure()
    component = Map.from_callable(
        source.carriers[0], target.carriers[0], lambda value: value
    )
    morphism = Morphism.from_components(
        source,
        target,
        (component,),
        preserved_operations=(source.signature.operations[0],),
        preserved_relations=(source.signature.relations[0],),
    )

    report = validate_homomorphism(morphism)
    assert type(report) is Proved
    assert report.operation_cases == 2
    assert report.relation_cases == 2
    inverse = Morphism.from_components(
        target,
        source,
        (
            Map.from_callable(
                target.carriers[0], source.carriers[0], lambda value: value
            ),
        ),
    )
    inverse_report = validate_inverse(morphism, inverse)
    assert type(inverse_report) is Proved
    assert inverse_report.inverse_cases == 4


def test_first_operation_counterexample_and_inconclusive_callable_are_typed() -> None:
    source = _structure()
    target = _structure(broken_target=True)
    component = Map.from_callable(
        source.carriers[0], target.carriers[0], lambda value: value
    )
    morphism = Morphism.from_components(
        source,
        target,
        (component,),
        preserved_operations=(source.signature.operations[0],),
    )
    report = validate_homomorphism(morphism)
    assert type(report) is Disproved
    assert report.counterexample.inputs == (0,)

    failing = Morphism.from_components(
        source,
        source,
        (
            Map.from_callable(
                source.carriers[0],
                source.carriers[0],
                lambda value: (_ for _ in ()).throw(ValueError("secret")),
            ),
        ),
        preserved_operations=(source.signature.operations[0],),
    )
    incomplete = validate_homomorphism(failing)
    assert type(incomplete) is Inconclusive
    assert "secret" not in incomplete.reason


def _carrier_only_structure(items: tuple[object, ...]) -> Structure:
    sort = Sort("value")
    signature = Signature((sort,))
    carrier = FiniteCarrier(items, sort=sort)
    return StructureBuilder(signature).with_carrier(sort, carrier).freeze()


def _nullary_structure(value: int) -> tuple[Structure, OperationSymbol]:
    sort = Sort("value")
    symbol = OperationSymbol("unit", (), sort)
    signature = Signature((sort,), operations=(symbol,))
    carrier = FiniteCarrier((0, 1), sort=sort)
    operation = Operation.from_table(symbol, ((sort, carrier),), (((), value),))
    structure = (
        StructureBuilder(signature)
        .with_carrier(sort, carrier)
        .with_operation(symbol, operation)
        .freeze()
    )
    return structure, symbol


def _relation_only_structure(true_tuples: tuple[tuple[int], ...]) -> Structure:
    sort = Sort("value")
    symbol = RelationSymbol("marked", (sort,))
    signature = Signature((sort,), relations=(symbol,))
    carrier = FiniteCarrier((0, 1), sort=sort)
    relation = Relation.from_tuples(symbol, ((sort, carrier),), true_tuples)
    return (
        StructureBuilder(signature)
        .with_carrier(sort, carrier)
        .with_relation(symbol, relation)
        .freeze()
    )


def test_full_signature_nullary_and_relation_direction_ignore_claim_metadata() -> None:
    source, _ = _nullary_structure(0)
    target, _ = _nullary_structure(0)
    component = Map.from_callable(source.carriers[0], target.carriers[0], lambda x: x)
    report = validate_homomorphism(
        Morphism.from_components(source, target, (component,))
    )
    assert type(report) is Proved
    assert report.counts.operation_expected == 1
    assert report.counts.operation_evaluated == 1
    assert report.counts.mapping_expected == 2
    assert report.scope.operation_symbols == source.signature.operations

    false_source = _relation_only_structure(())
    true_target = _relation_only_structure(((0,),))
    relation_map = Map.from_callable(
        false_source.carriers[0], true_target.carriers[0], lambda x: x
    )
    direction_report = validate_homomorphism(
        Morphism.from_components(false_source, true_target, (relation_map,))
    )
    assert type(direction_report) is Proved
    assert direction_report.relation_cases == 2


def test_many_sorted_components_do_not_dispatch_on_overlapping_raw_values() -> None:
    left = Sort("left")
    right = Sort("right")
    operation_symbol = OperationSymbol("right_projection", (left, right), right)
    relation_symbol = RelationSymbol("aligned", (left, right))
    signature = Signature(
        (left, right), operations=(operation_symbol,), relations=(relation_symbol,)
    )

    def build(relation_tuples: tuple[tuple[int, int], ...]) -> Structure:
        left_carrier = FiniteCarrier((0, 1), sort=left)
        right_carrier = FiniteCarrier((0, 1), sort=right)
        operation = Operation.from_table(
            operation_symbol,
            ((left, left_carrier), (right, right_carrier)),
            (((0, 0), 0), ((0, 1), 1), ((1, 0), 0), ((1, 1), 1)),
        )
        relation = Relation.from_tuples(
            relation_symbol,
            ((left, left_carrier), (right, right_carrier)),
            relation_tuples,
        )
        return (
            StructureBuilder(signature)
            .with_carrier(left, left_carrier)
            .with_carrier(right, right_carrier)
            .with_operation(operation_symbol, operation)
            .with_relation(relation_symbol, relation)
            .freeze()
        )

    source = build(((0, 0),))
    target = build(((0, 1),))
    morphism = Morphism.from_components(
        source,
        target,
        (
            Map.from_callable(source.carriers[0], target.carriers[0], lambda x: x),
            Map.from_callable(source.carriers[1], target.carriers[1], lambda x: 1 - x),
        ),
    )
    report = validate_homomorphism(morphism)
    assert type(report) is Proved
    assert report.operation_cases == 4
    assert report.relation_cases == 4


def test_invalid_component_and_unhashable_carriers_are_checked_before_symbols() -> None:
    source = _carrier_only_structure((0, 1))
    target = _carrier_only_structure((0, 1))
    invalid = Morphism.from_components(
        source,
        target,
        (Map.from_callable(source.carriers[0], target.carriers[0], lambda _: 2),),
    )
    report = validate_homomorphism(invalid)
    assert type(report) is Disproved
    assert report.counterexample.kind == "invalid_component_image"
    assert report.counts.mapping_evaluated == 1

    unhashable_source = _carrier_only_structure(([], [1]))
    unhashable_target = _carrier_only_structure(([], [1]))
    unhashable = Morphism.from_components(
        unhashable_source,
        unhashable_target,
        (
            Map.from_callable(
                unhashable_source.carriers[0],
                unhashable_target.carriers[0],
                lambda value: unhashable_target.carriers[0].items[
                    unhashable_source.carriers[0].index(value)
                ],
            ),
        ),
    )
    assert type(validate_homomorphism(unhashable)) is Proved


def test_partiality_never_proves_and_later_disproof_wins() -> None:
    sort = Sort("value")
    symbol = OperationSymbol("partial", (sort,), sort)
    signature = Signature((sort,), operations=(symbol,))
    source_carrier = FiniteCarrier((0, 1), sort=sort)
    target_carrier = FiniteCarrier((0, 1), sort=sort)
    source_operation = PartialOperation.from_callable(
        symbol,
        ((sort, source_carrier),),
        lambda x: Undefined("outside-domain") if x == 0 else Defined(1),
    )
    target_operation = Operation.from_table(
        symbol,
        ((sort, target_carrier),),
        (((0,), 0), ((1,), 0)),
    )
    source = (
        StructureBuilder(signature)
        .with_carrier(sort, source_carrier)
        .with_operation(symbol, source_operation)
        .freeze()
    )
    target = (
        StructureBuilder(signature)
        .with_carrier(sort, target_carrier)
        .with_operation(symbol, target_operation)
        .freeze()
    )
    identity = Map.from_callable(source_carrier, target_carrier, lambda x: x)
    later_disproof = validate_homomorphism(
        Morphism.from_components(source, target, (identity,))
    )
    assert type(later_disproof) is Disproved
    assert later_disproof.counterexample.inputs == (1,)

    for outcome in (Failed("failed"), Indeterminate("unknown")):
        unknown_operation = Operation.from_callable(
            symbol,
            ((sort, source_carrier),),
            lambda _x, result=outcome: result,
        )
        unknown_source = (
            StructureBuilder(signature)
            .with_carrier(sort, source_carrier)
            .with_operation(symbol, unknown_operation)
            .freeze()
        )
        unknown_target = (
            StructureBuilder(signature)
            .with_carrier(sort, target_carrier)
            .with_operation(symbol, target_operation)
            .freeze()
        )
        unknown_map = Map.from_callable(source_carrier, target_carrier, lambda x: x)
        report = validate_homomorphism(
            Morphism.from_components(unknown_source, unknown_target, (unknown_map,))
        )
        assert type(report) is Inconclusive


def test_inverse_bounds_and_combined_isomorphism_share_component_sessions() -> None:
    source = _carrier_only_structure((0, 1))
    target = _carrier_only_structure((0, 1, 2))
    forward = Morphism.from_components(
        source,
        target,
        (Map.from_callable(source.carriers[0], target.carriers[0], lambda x: x),),
    )
    backward = Morphism.from_components(
        target,
        source,
        (Map.from_callable(target.carriers[0], source.carriers[0], lambda x: x % 2),),
    )
    inverse_report = validate_inverse(forward, backward)
    assert type(inverse_report) is Disproved
    assert inverse_report.counts.inverse_expected == 5

    left, _ = _nullary_structure(0)
    right, _ = _nullary_structure(0)
    calls = {"forward": 0, "backward": 0}

    def forward_function(value: int) -> int:
        calls["forward"] += 1
        return value

    def backward_function(value: int) -> int:
        calls["backward"] += 1
        return value

    map_forward = Morphism.from_components(
        left,
        right,
        (Map.from_callable(left.carriers[0], right.carriers[0], forward_function),),
    )
    map_backward = Morphism.from_components(
        right,
        left,
        (Map.from_callable(right.carriers[0], left.carriers[0], backward_function),),
    )
    report = validate_isomorphism(map_forward, map_backward)
    assert type(report) is Proved
    assert calls == {"forward": 2, "backward": 2}
    assert report.scope.source is left
    assert report.scope.target is right
    assert report.scope.source_carrier_sizes == (2,)
    assert report.scope.target_carrier_sizes == (2,)


def test_public_validation_records_reject_malformed_direct_construction() -> None:
    with pytest.raises(TypeError, match="operation_symbols"):
        ValidationScope((cast(OperationSymbol, object()),), ())
    with pytest.raises(ValueError, match="exceeds"):
        ValidationCounts(0, 1, 0, 0)
    with pytest.raises(ValueError, match="exhaustive"):
        Proved(ValidationScope((), ()), ValidationCounts(1, 0, 0, 0))


def test_zero_sorted_unique_morphism_and_high_arity_singleton_operation() -> None:
    empty_source = StructureBuilder(Signature(())).freeze()
    empty_target = StructureBuilder(Signature(())).freeze()
    empty = Morphism.from_components(empty_source, empty_target, ())
    empty_report = validate_homomorphism(empty)
    assert type(empty_report) is Proved
    assert empty_report.counts.mapping_expected == 0
    assert empty_report.scope.source is empty_source
    assert empty_report.scope.target is empty_target

    sort = Sort("singleton")
    symbol = OperationSymbol("arity257", (sort,) * 257, sort)
    signature = Signature((sort,), operations=(symbol,))
    source_carrier = FiniteCarrier((0,), sort=sort)
    target_carrier = FiniteCarrier((0,), sort=sort)
    input_tuple = (0,) * 257
    source = (
        StructureBuilder(signature)
        .with_carrier(sort, source_carrier)
        .with_operation(
            symbol,
            Operation.from_table(
                symbol, ((sort, source_carrier),), ((input_tuple, 0),)
            ),
        )
        .freeze()
    )
    target = (
        StructureBuilder(signature)
        .with_carrier(sort, target_carrier)
        .with_operation(
            symbol,
            Operation.from_table(
                symbol, ((sort, target_carrier),), ((input_tuple, 0),)
            ),
        )
        .freeze()
    )
    high_arity = Morphism.from_components(
        source,
        target,
        (Map.from_callable(source_carrier, target_carrier, lambda x: x),),
    )
    report = validate_homomorphism(high_arity)
    assert type(report) is Proved
    assert report.counts.operation_expected == 1
    assert report.counts.operation_evaluated == 1
    assert report.scope.source is source
    assert report.scope.target is target
    assert report.scope.source_carrier_sizes == (1,)
    assert report.scope.target_carrier_sizes == (1,)
    assert report.scope.algorithm == "finite-structure-map-v1"
    assert "declaration order" in report.scope.enumeration_order


def test_isomorphism_endpoint_preflight_and_safe_evidence_reprs() -> None:
    left, _ = _nullary_structure(0)
    right, _ = _nullary_structure(0)
    calls = {"left": 0, "right": 0}

    def left_function(value: int) -> int:
        calls["left"] += 1
        return value

    def right_function(value: int) -> int:
        calls["right"] += 1
        return value

    forward = Morphism.from_components(
        left,
        right,
        (Map.from_callable(left.carriers[0], right.carriers[0], left_function),),
    )
    malformed_inverse = Morphism.from_components(
        left,
        right,
        (Map.from_callable(left.carriers[0], right.carriers[0], right_function),),
    )
    mismatch = validate_isomorphism(forward, malformed_inverse)
    assert type(mismatch) is Inconclusive
    assert calls == {"left": 0, "right": 0}
    assert mismatch.counts.inverse_expected == 4
    assert mismatch.counts.inverse_evaluated == 0
    assert mismatch.counts.mapping_expected == 4
    assert mismatch.counts.mapping_evaluated == 0

    inverse = Morphism.from_components(
        right,
        left,
        (Map.from_callable(right.carriers[0], left.carriers[0], lambda _x: 0),),
    )
    inverse_failure = validate_inverse(forward, inverse)
    assert type(inverse_failure) is Disproved
    assert inverse_failure.counts.inverse_expected == 4
    assert inverse_failure.counts.inverse_evaluated == 2
    assert inverse_failure.counts.mapping_expected == 4
    assert inverse_failure.counts.mapping_evaluated == 4

    class NoisyWitness:
        def __repr__(self) -> str:
            return "sensitive-witness-0xabcdef"

    witness = Counterexample("operation", None, None, (NoisyWitness(),))
    disproved = Disproved(
        witness, ValidationScope((), ()), ValidationCounts(0, 0, 0, 0)
    )
    inconclusive = Inconclusive(
        "safe reason", ValidationScope((), ()), ValidationCounts(0, 0, 0, 0)
    )
    for value in (witness, disproved, inconclusive):
        assert "sensitive-witness" not in repr(value)
        assert "0x" not in repr(value)
