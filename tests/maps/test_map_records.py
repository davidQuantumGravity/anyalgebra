"""Contract tests for explicit, record-only typed map declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
import traceback
from typing import cast

import pytest

from anyalgebra.maps.core import (
    Map,
    MapDefinitionError,
    LinearMap,
    Morphism,
    MultilinearMap,
)
from anyalgebra.fixtures.composition import quaternion_fixture
from anyalgebra.core.domains import ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder


def _module(label: str) -> FreeModule:
    return FreeModule(ZZ(), Basis((label,), coefficient_domain=ZZ()))


def _structure() -> tuple[Structure, OperationSymbol, RelationSymbol]:
    sort = Sort("value")
    operation = OperationSymbol("unit", (), sort)
    relation = RelationSymbol("valid", (sort,))
    signature = Signature((sort,))
    carrier = FiniteCarrier((0,), sort=sort)
    structure = StructureBuilder(signature).with_carrier(sort, carrier).freeze()
    return structure, operation, relation


def _identity(value: object) -> object:
    return value


def _empty_structure() -> Structure:
    return StructureBuilder(Signature(())).freeze()


def _two_sorted_structure() -> Structure:
    first = Sort("first")
    second = Sort("second")
    signature = Signature((first, second))
    return (
        StructureBuilder(signature)
        .with_carrier(first, FiniteCarrier((0,), sort=first))
        .with_carrier(second, FiniteCarrier((0,), sort=second))
        .freeze()
    )


def test_map_and_linear_map_retain_literal_endpoints_and_inert_guarantees() -> None:
    source = _module("e")
    target = _module("f")
    plain = Map.from_callable(source, target, _identity, guarantees=("exact",))
    linear = LinearMap.from_callable(source, target, _identity, guarantees=("linear",))

    assert type(plain) is Map
    assert type(linear) is LinearMap
    assert plain.source is source
    assert plain.target is target
    assert plain.guarantees == ("exact",)
    assert linear.source is source
    assert linear.target is target
    assert linear.guarantees == ("linear",)
    assert not hasattr(plain, "apply")
    assert not hasattr(linear, "apply")


def test_map_does_not_merge_same_looking_distinct_parents_or_convert() -> None:
    source = _module("e")
    same_looking = _module("e")
    record = Map.from_callable(source, same_looking, _identity)

    assert source == same_looking
    assert source is not same_looking
    assert record.source is source
    assert record.target is same_looking
    assert not hasattr(record, "convert")
    assert not hasattr(record, "coerce")


def test_generic_map_accepts_current_finite_multilinear_structure_parents() -> None:
    source = quaternion_fixture()
    target = quaternion_fixture()
    record = Map.from_callable(source, target, _identity)

    assert record.source is source
    assert record.target is target
    assert source is not target


def test_multilinear_map_retains_order_nullary_and_high_arity_sources() -> None:
    first = _module("a")
    second = _module("b")
    target = _module("out")
    ordered = MultilinearMap.from_callable((first, second), target, _identity)
    reversed_record = MultilinearMap.from_callable((second, first), target, _identity)
    nullary = MultilinearMap.from_callable((), target, _identity)
    high_arity = MultilinearMap.from_callable((first,) * 257, target, _identity)

    assert ordered.sources == (first, second)
    assert reversed_record.sources == (second, first)
    assert ordered.sources != reversed_record.sources
    assert nullary.sources == ()
    assert len(high_arity.sources) == 257
    assert high_arity.sources[0] is first


def test_map_metadata_preflights_and_bounds_iterables_without_calling_function() -> (
    None
):
    source = _module("e")
    target = _module("f")
    function = _TrackedCallable()
    untouched = _TouchedGuarantees()
    with pytest.raises(MapDefinitionError, match="source"):
        Map.from_callable(
            cast(FreeModule, object()), target, function, guarantees=untouched
        )
    assert not untouched.touched
    assert function.calls == 0
    overlong = ("g",) * 65
    with pytest.raises(MapDefinitionError, match="maximum"):
        Map.from_callable(source, target, function, guarantees=overlong)
    with pytest.raises(MapDefinitionError, match="maximum"):
        MultilinearMap.from_callable((source,) * 513, target, function)
    assert function.calls == 0


def test_invalid_callable_preflights_before_all_user_iterables() -> None:
    source = _module("e")
    target = _module("f")
    map_guarantees = _TouchedGuarantees()
    with pytest.raises(MapDefinitionError, match="callable"):
        Map.from_callable(source, target, object(), guarantees=map_guarantees)
    assert not map_guarantees.touched

    multilinear_sources = _TouchedParents(source)
    multilinear_guarantees = _TouchedGuarantees()
    with pytest.raises(MapDefinitionError, match="callable"):
        MultilinearMap.from_callable(
            multilinear_sources,
            target,
            object(),
            guarantees=multilinear_guarantees,
        )
    assert not multilinear_sources.touched
    assert not multilinear_guarantees.touched

    structure, operation, relation = _structure()
    morphism_guarantees = _TouchedGuarantees()
    operations = _TouchedSymbols(operation)
    relations = _TouchedSymbols(relation)
    with pytest.raises(MapDefinitionError, match="callable"):
        Morphism.from_callable(
            structure,
            structure,
            object(),
            guarantees=morphism_guarantees,
            preserved_operations=operations,
            preserved_relations=relations,
        )
    assert not morphism_guarantees.touched
    assert not operations.touched
    assert not relations.touched


def test_morphism_callable_preflights_signatures_and_sort_counts_before_metadata() -> (
    None
):
    one_sorted, _, _ = _structure()
    cases = (
        (_empty_structure(), _empty_structure(), "one-sorted"),
        (_two_sorted_structure(), _two_sorted_structure(), "one-sorted"),
        (one_sorted, _empty_structure(), "same structural signature"),
        (_empty_structure(), one_sorted, "same structural signature"),
    )
    for source, target, message in cases:
        function = _TrackedCallable()
        metadata = _TouchedGuarantees()
        with pytest.raises(MapDefinitionError, match=message):
            Morphism.from_callable(source, target, function, guarantees=metadata)
        assert function.calls == 0
        assert not metadata.touched


def test_rejects_malformed_parent_callable_and_guarantee_metadata() -> None:
    source = _module("e")
    target = _module("f")
    with pytest.raises(MapDefinitionError, match="declared parent"):
        Map.from_callable(source, object(), _identity)
    with pytest.raises(MapDefinitionError, match="callable"):
        Map.from_callable(source, target, cast(Callable[..., object], object()))
    with pytest.raises(MapDefinitionError, match="guarantees"):
        Map.from_callable(source, target, _identity, guarantees="exact")
    with pytest.raises(MapDefinitionError, match="exact built-in str"):
        Map.from_callable(source, target, _identity, guarantees=("exact", 1))
    with pytest.raises(MapDefinitionError, match="duplicate"):
        Map.from_callable(source, target, _identity, guarantees=("exact", "exact"))
    with pytest.raises(MapDefinitionError, match="sources"):
        MultilinearMap.from_callable(
            cast(tuple[object, ...], object()), target, _identity
        )


def test_linear_and_multilinear_records_require_exact_module_endpoints() -> None:
    source = _module("e")
    target = _module("f")
    with pytest.raises(MapDefinitionError, match="exact FreeModule"):
        LinearMap.from_callable(ZZ(), target, _identity)
    with pytest.raises(MapDefinitionError, match="exact FreeModule"):
        LinearMap.from_callable(source, ZZ(), _identity)
    with pytest.raises(MapDefinitionError, match="exact FreeModule"):
        MultilinearMap.from_callable((ZZ(),), target, _identity)
    with pytest.raises(MapDefinitionError, match="exact FreeModule"):
        MultilinearMap.from_callable((source,), ZZ(), _identity)


def test_morphism_is_the_documented_structure_map_record_with_inert_preservation() -> (
    None
):
    source, operation, relation = _structure()
    target, _, _ = _structure()
    record = Morphism.from_callable(
        source,
        target,
        _identity,
        guarantees=("candidate",),
        preserved_operations=(operation,),
        preserved_relations=(relation,),
    )

    assert type(record) is Morphism
    assert record.source is source
    assert record.target is target
    assert record.preserved_operations == (operation,)
    assert record.preserved_relations == (relation,)
    assert not hasattr(record, "validate")
    assert not hasattr(record, "is_homomorphism")
    with pytest.raises(MapDefinitionError, match="exact Structure"):
        Morphism.from_callable(_module("e"), target, _identity)
    with pytest.raises(MapDefinitionError, match="OperationSymbol"):
        Morphism.from_callable(
            source, target, _identity, preserved_operations=(object(),)
        )
    with pytest.raises(MapDefinitionError, match="RelationSymbol"):
        Morphism.from_callable(
            source, target, _identity, preserved_relations=(object(),)
        )


def test_records_are_factory_only_frozen_unhashable_identity_only_and_safe_repr() -> (
    None
):
    source = _module("e")
    target = _module("f")
    first = Map.from_callable(source, target, _identity)
    second = Map.from_callable(source, target, _identity)
    linear = LinearMap.from_callable(source, target, _identity)
    second_linear = LinearMap.from_callable(source, target, _identity)
    multilinear = MultilinearMap.from_callable((source,), target, _identity)
    structure, _, _ = _structure()
    morphism = Morphism.from_callable(structure, structure, _identity)
    second_morphism = Morphism.from_callable(structure, structure, _identity)

    assert first != second
    assert linear != second_linear
    assert morphism != second_morphism
    for record in (first, linear, multilinear, morphism):
        assert not hasattr(record, "__dict__")
        assert "0x" not in repr(record)
        with pytest.raises(TypeError, match="unhashable"):
            hash(record)
    with pytest.raises(FrozenInstanceError):
        first.source = target  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        linear.target = source  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        multilinear.target = source  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        morphism.target = structure  # type: ignore[misc]
    with pytest.raises(MapDefinitionError, match="from_callable"):
        Map(source, target, _identity, ())
    with pytest.raises(MapDefinitionError, match="from_callable"):
        LinearMap(source, target, _identity, ())
    with pytest.raises(MapDefinitionError, match="from_callable"):
        MultilinearMap((source,), target, _identity, ())
    with pytest.raises(MapDefinitionError, match="from_callable"):
        Morphism(structure, structure, _identity, ())


def test_metadata_bounds_duplicate_symbols_and_iterator_sanitation() -> None:
    source = _module("e")
    target = _module("f")
    guarantees = tuple(f"g{index}" for index in range(64))
    record = Map.from_callable(source, target, _identity, guarantees=guarantees)
    assert record.guarantees == guarantees
    overlong_guarantees = _CountingRepeater("g")
    with pytest.raises(MapDefinitionError, match="maximum"):
        Map.from_callable(source, target, _identity, guarantees=overlong_guarantees)
    assert overlong_guarantees.pulls == 65

    sources = _CountingRepeater(source)
    with pytest.raises(MapDefinitionError, match="maximum"):
        MultilinearMap.from_callable(sources, target, _identity)
    assert sources.pulls == 513
    accepted = MultilinearMap.from_callable((source,) * 512, target, _identity)
    assert len(accepted.sources) == 512

    structure, operation, relation = _structure()
    operations = tuple(
        OperationSymbol(f"operation{index}", (), Sort("value")) for index in range(64)
    )
    relations = tuple(
        RelationSymbol(f"relation{index}", (Sort("value"),)) for index in range(64)
    )
    morphism = Morphism.from_callable(
        structure,
        structure,
        _identity,
        preserved_operations=operations,
        preserved_relations=relations,
    )
    assert len(morphism.preserved_operations) == 64
    assert len(morphism.preserved_relations) == 64
    overlong_operations = _CountingRepeater(operation)
    with pytest.raises(MapDefinitionError, match="maximum"):
        Morphism.from_callable(
            structure, structure, _identity, preserved_operations=overlong_operations
        )
    assert overlong_operations.pulls == 65
    overlong_relations = _CountingRepeater(relation)
    with pytest.raises(MapDefinitionError, match="maximum"):
        Morphism.from_callable(
            structure, structure, _identity, preserved_relations=overlong_relations
        )
    assert overlong_relations.pulls == 65
    with pytest.raises(MapDefinitionError, match="duplicate"):
        Morphism.from_callable(
            structure,
            structure,
            _identity,
            preserved_operations=(operation, operation),
        )
    with pytest.raises(MapDefinitionError, match="duplicate"):
        Morphism.from_callable(
            structure,
            structure,
            _identity,
            preserved_relations=(relation, relation),
        )

    def broken_guarantees() -> Iterator[str]:
        yield "exact"
        raise RuntimeError("secret map metadata payload")

    with pytest.raises(MapDefinitionError, match="iterated") as caught:
        Map.from_callable(source, target, _identity, guarantees=broken_guarantees())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret map metadata payload" not in "".join(
        traceback.format_exception(caught.value)
    )


class _TrackedCallable:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value: object) -> object:
        self.calls += 1
        return value


class _TouchedGuarantees:
    def __init__(self) -> None:
        self.touched = False

    def __iter__(self) -> Iterator[str]:
        self.touched = True
        yield "exact"


class _TouchedParents:
    def __init__(self, parent: FreeModule) -> None:
        self.parent = parent
        self.touched = False

    def __iter__(self) -> Iterator[FreeModule]:
        self.touched = True
        yield self.parent


class _TouchedSymbols:
    def __init__(self, symbol: OperationSymbol | RelationSymbol) -> None:
        self.symbol = symbol
        self.touched = False

    def __iter__(self) -> Iterator[OperationSymbol | RelationSymbol]:
        self.touched = True
        yield self.symbol


class _CountingRepeater:
    def __init__(self, value: object) -> None:
        self.value = value
        self.pulls = 0

    def __iter__(self) -> _CountingRepeater:
        return self

    def __next__(self) -> object:
        self.pulls += 1
        return self.value
