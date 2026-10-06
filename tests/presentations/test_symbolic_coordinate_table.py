"""Contract tests for explicit symbolic/coordinate/table presentation adapters."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import FrozenInstanceError
from itertools import product
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures.composition import octonion_fixture
from anyalgebra.presentations.adapters import (
    BasisExpression,
    PresentationAdapterBundle,
    PresentationAdapterError,
)
from anyalgebra.presentations.convert import (
    ConversionAmbiguityError,
    RouteStep,
    convert,
    plan_conversion,
)
from anyalgebra.presentations.graph import (
    Conversion,
    ConversionDefinitionError,
    ConversionGraph,
)


def _module(rank: int, *, qq: bool = False) -> FreeModule:
    domain = QQ() if qq else ZZ()
    return FreeModule(
        domain,
        Basis(tuple(f"e{index}" for index in range(rank)), coefficient_domain=domain),
    )


def _table(
    module: FreeModule, arity: int, cells: tuple[object, ...]
) -> BasisProductTable:
    return BasisProductTable.from_cells(module, arity, cells)  # type: ignore[arg-type]


def test_symbolic_coordinates_canonicalize_and_round_trip_direct_and_v52() -> None:
    module = _module(3)
    bundle = PresentationAdapterBundle.for_module(module, "fixture.vector", 2)
    expression = BasisExpression.from_terms(
        module,
        (("e2", ZZ().element(3)), ("e0", 1), ("e1", 0)),
    )
    assert tuple(label for label, _ in expression.terms) == ("e0", "e2")
    coordinates = bundle.coordinates_from_symbolic(expression)
    assert coordinates.parent is module
    assert tuple(coordinates.coordinates().items()) == (
        (0, ZZ().element(1)),
        (2, ZZ().element(3)),
    )
    assert bundle.symbolic_from_coordinates(coordinates) == expression
    converted = convert(
        expression,
        bundle.coordinate_kind,
        graph=bundle.graph,
        source_kind=bundle.symbolic_kind,
    )
    assert converted.value == coordinates
    assert converted.route_witness == (
        ("fixture.vector.symbolic-to-coordinate", "forward"),
    )
    assert converted.guarantees == ("exact", "basis-order-preserving")
    assert converted.round_trip_ids == ("fixture.vector.symbolic-coordinate",)
    inverse = convert(
        coordinates,
        bundle.symbolic_kind,
        graph=bundle.graph,
        source_kind=bundle.coordinate_kind,
    )
    assert inverse.value == expression
    assert inverse.validity_domains == ("literal-module:fixture.vector",)


def test_symbolic_rank_zero_qq_and_negative_parent_boundaries() -> None:
    zero_module = _module(0)
    zero = BasisExpression.from_terms(zero_module, ())
    assert zero.terms == ()
    assert zero.to_sparse() == zero_module.zero()
    qq_module = _module(1, qq=True)
    expression = BasisExpression.from_terms(qq_module, (("e0", QQ().element(1)),))
    assert expression.to_sparse().parent is qq_module
    with pytest.raises(PresentationAdapterError, match="not present"):
        BasisExpression.from_terms(qq_module, (("missing", 1),))
    with pytest.raises(PresentationAdapterError, match="duplicate"):
        BasisExpression.from_terms(qq_module, (("e0", 1), ("e0", 2)))
    with pytest.raises(PresentationAdapterError, match="normalized"):
        BasisExpression.from_terms(qq_module, (("e0", ZZ().element(1)),))
    with pytest.raises(PresentationAdapterError, match="raw string"):
        BasisExpression.from_terms(qq_module, "e0")
    with pytest.raises(PresentationAdapterError, match="exact pair"):
        BasisExpression.from_terms(qq_module, (["e0", 1],))
    with pytest.raises(PresentationAdapterError, match="exact built-in str"):
        BasisExpression.from_terms(qq_module, ((StringSubclass("e0"), 1),))
    with pytest.raises(PresentationAdapterError, match="normalized"):
        BasisExpression.from_terms(qq_module, (("e0", object()),))
    with pytest.raises(PresentationAdapterError, match="exact SparseElement"):
        BasisExpression.from_sparse(qq_module, object())
    with pytest.raises(PresentationAdapterError, match="literal module"):
        BasisExpression.from_sparse(_module(1, qq=True), expression.to_sparse())


def test_symbolic_snapshots_are_bounded_safe_and_factory_records_are_closed() -> None:
    module = _module(1)

    def hostile() -> Iterator[tuple[str, int]]:
        yield ("e0", 1)
        raise RuntimeError("secret iterator payload")

    with pytest.raises(
        PresentationAdapterError, match="could not be iterated"
    ) as caught:
        BasisExpression.from_terms(module, hostile())
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    for hostile_source in (HostileItemsMapping(), HostileIter(), HostileNext()):
        with pytest.raises(
            PresentationAdapterError, match="could not be iterated"
        ) as caught:
            BasisExpression.from_terms(module, hostile_source)
        assert "secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    with pytest.raises(PresentationAdapterError, match="maximum 512 symbolic terms"):
        BasisExpression.from_terms(module, (("e0", 1),) * 513)
    expression = BasisExpression.from_terms(module, (("e0", 1),))
    bundle = PresentationAdapterBundle.for_module(module, "fixture.closed", 1)
    for record in (expression, bundle):
        assert not hasattr(record, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(record)
        assert "0x" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        expression.terms = ()  # type: ignore[misc]
    with pytest.raises(PresentationAdapterError, match="from_terms"):
        BasisExpression(module, ())
    with pytest.raises(PresentationAdapterError, match="for_module"):
        PresentationAdapterBundle(module, "bad", 1)


def test_symbolic_512_term_bound_and_canonical_equality() -> None:
    module = _module(512)
    terms = tuple((f"e{index}", 1) for index in reversed(range(512)))
    expression = BasisExpression.from_terms(module, terms)
    assert len(expression.terms) == 512
    assert expression.terms[0][0] == "e0"
    with pytest.raises(PresentationAdapterError, match="maximum 512 symbolic terms"):
        BasisExpression.from_terms(
            _module(513), tuple((f"e{index}", 1) for index in range(513))
        )
    zero = BasisExpression.from_terms(module, (("e0", 0),))
    assert zero == BasisExpression.from_terms(module, ())
    assert expression == BasisExpression.from_sparse(module, expression.to_sparse())
    assert expression != BasisExpression.from_terms(_module(512), terms)


def test_adapter_factory_boundaries_kinds_and_stable_namespace_semantics() -> None:
    module = _module(1)
    for namespace in ("", " ", StringSubclass("fixture.subclass")):
        with pytest.raises(PresentationAdapterError, match="namespace"):
            PresentationAdapterBundle.for_module(module, namespace, 1)
    for arity in (True, -1, "1"):
        with pytest.raises(PresentationAdapterError, match="arity"):
            PresentationAdapterBundle.for_module(module, "fixture.arity", arity)
    with pytest.raises(PresentationAdapterError, match="exact FreeModule"):
        PresentationAdapterBundle.for_module(object(), "fixture.module", 1)
    bundle = PresentationAdapterBundle.for_module(module, "fixture.metadata", 2)
    replica = PresentationAdapterBundle.for_module(module, "fixture.metadata", 2)
    assert (
        bundle.symbolic_kind,
        bundle.coordinate_kind,
        bundle.table_kind,
        bundle.structure_kind,
    ) == (
        replica.symbolic_kind,
        replica.coordinate_kind,
        replica.table_kind,
        replica.structure_kind,
    )
    assert tuple(
        kind.id
        for kind in (
            bundle.symbolic_kind,
            bundle.coordinate_kind,
            bundle.table_kind,
            bundle.structure_kind,
        )
    ) == (
        "fixture.metadata.symbolic",
        "fixture.metadata.coordinate",
        "fixture.metadata.table",
        "fixture.metadata.structure",
    )
    assert (
        bundle.symbolic_to_coordinate.stable_id
        == "fixture.metadata.symbolic-to-coordinate"
    )
    assert bundle.table_to_structure.stable_id == "fixture.metadata.table-to-structure"
    for conversion, round_trip, validity in (
        (
            bundle.symbolic_to_coordinate,
            "fixture.metadata.symbolic-coordinate",
            "literal-module:fixture.metadata",
        ),
        (
            bundle.table_to_structure,
            "fixture.metadata.table-structure",
            "literal-module:fixture.metadata;arity:2",
        ),
    ):
        assert conversion.inverse is not None
        assert conversion.guarantees == ("exact", "basis-order-preserving")
        assert conversion.round_trip_id == round_trip
        assert conversion.validity_domain == validity

    class ExpressionSubclass(BasisExpression):
        pass

    class BundleSubclass(PresentationAdapterBundle):
        pass

    with pytest.raises(PresentationAdapterError, match="exact BasisExpression"):
        ExpressionSubclass.from_terms(module, ())
    with pytest.raises(
        PresentationAdapterError, match="exact PresentationAdapterBundle"
    ):
        BundleSubclass.for_module(module, "fixture.subclass", 1)


@pytest.mark.parametrize("arity", (0, 1, 3))
def test_table_structure_table_round_trips_arbitrary_arity(arity: int) -> None:
    module = _module(2)
    cells = tuple(
        module.element({index % 2: index + 1}) for index in range(module.rank**arity)
    )
    table = _table(module, arity, cells)
    bundle = PresentationAdapterBundle.for_module(
        module, f"fixture.table.{arity}", arity
    )
    structure = bundle.structure_from_table(table)
    reconstructed = bundle.table_from_structure(structure)
    assert reconstructed.module is module
    assert reconstructed.arity == arity
    assert reconstructed.cells == table.cells
    for inputs in product(range(module.rank), repeat=arity):
        assert (
            structure.evaluate_basis(*inputs)
            == reconstructed.cells[
                tuple(product(range(module.rank), repeat=arity)).index(inputs)
            ]
        )
    converted = convert(
        table,
        bundle.structure_kind,
        graph=bundle.graph,
        source_kind=bundle.table_kind,
    )
    assert converted.value == structure
    restored = convert(
        structure,
        bundle.table_kind,
        graph=bundle.graph,
        source_kind=bundle.structure_kind,
    )
    assert restored.value == table
    assert restored.round_trip_ids == (f"fixture.table.{arity}.table-structure",)


def test_rank_zero_and_257_ary_rank_one_tables_remain_iterative() -> None:
    rank_zero = _module(0)
    bundle_zero = PresentationAdapterBundle.for_module(rank_zero, "fixture.zero", 0)
    nullary = _table(rank_zero, 0, (rank_zero.zero(),))
    assert (
        bundle_zero.table_from_structure(bundle_zero.structure_from_table(nullary))
        == nullary
    )
    unary = _table(rank_zero, 1, ())
    unary_bundle = PresentationAdapterBundle.for_module(
        rank_zero, "fixture.zero.unary", 1
    )
    assert (
        unary_bundle.table_from_structure(unary_bundle.structure_from_table(unary))
        == unary
    )
    singleton = _module(1)
    high = _table(singleton, 257, (singleton.element({0: 1}),))
    high_bundle = PresentationAdapterBundle.for_module(singleton, "fixture.high", 257)
    assert (
        high_bundle.table_from_structure(high_bundle.structure_from_table(high)) == high
    )


def test_octonion_table_and_structure_cells_round_trip() -> None:
    structure = octonion_fixture()
    module = structure.module
    original = BasisProductTable.from_cells(
        module,
        2,
        tuple(
            structure.evaluate_basis(*inputs) for inputs in product(range(8), repeat=2)
        ),
    )
    bundle = PresentationAdapterBundle.for_module(module, "fixture.octonion", 2)
    rebuilt = bundle.table_from_structure(bundle.structure_from_table(original))
    assert rebuilt.cells == original.cells
    table = cast(
        BasisProductTable,
        convert(
            structure,
            bundle.table_kind,
            graph=bundle.graph,
            source_kind=bundle.structure_kind,
        ).value,
    )
    assert type(table) is BasisProductTable
    rebuilt_structure = cast(
        FiniteMultilinearStructure,
        convert(
            table,
            bundle.structure_kind,
            graph=bundle.graph,
            source_kind=bundle.table_kind,
        ).value,
    )
    assert type(rebuilt_structure) is FiniteMultilinearStructure
    assert rebuilt_structure.name is None
    for inputs in product(range(module.rank), repeat=2):
        assert rebuilt_structure.evaluate_basis(*inputs) == structure.evaluate_basis(
            *inputs
        )


def test_adapter_preflight_coexisting_namespaces_and_persistent_registration() -> None:
    left_module = _module(1)
    right_module = _module(1)
    left = PresentationAdapterBundle.for_module(left_module, "fixture.left", 1)
    right = PresentationAdapterBundle.for_module(right_module, "fixture.right", 1)
    assert left.symbolic_kind != right.symbolic_kind
    graph = left.register_into(ConversionGraph.empty())
    expanded = right.register_into(graph)
    assert len(graph.edges) == 2
    assert len(expanded.edges) == 4
    assert plan_conversion(left.symbolic_kind, left.coordinate_kind, graph=expanded)
    with pytest.raises(PresentationAdapterError, match="literal module"):
        left.coordinates_from_symbolic(
            BasisExpression.from_terms(right_module, (("e0", 1),))
        )
    with pytest.raises(PresentationAdapterError, match="exact ConversionGraph"):
        left.register_into(object())
    foreign_table = _table(right_module, 1, (right_module.zero(),))
    with pytest.raises(PresentationAdapterError, match="literal module"):
        left.structure_from_table(foreign_table)
    with pytest.raises(PresentationAdapterError, match="declared adapter arity"):
        left.structure_from_table(_table(left_module, 0, (left_module.zero(),)))
    with pytest.raises(
        PresentationAdapterError, match="exact FiniteMultilinearStructure"
    ):
        left.table_from_structure(object())
    with pytest.raises(PresentationAdapterError, match="exact BasisProductTable"):
        left.structure_from_table(object())
    right_structure = right.structure_from_table(foreign_table)
    with pytest.raises(PresentationAdapterError, match="literal module"):
        left.table_from_structure(right_structure)
    left_nullary = PresentationAdapterBundle.for_module(
        left_module, "fixture.left.nullary", 0
    )
    nullary_structure = left_nullary.structure_from_table(
        _table(left_module, 0, (left_module.zero(),))
    )
    with pytest.raises(PresentationAdapterError, match="declared adapter arity"):
        left.table_from_structure(nullary_structure)
    with pytest.raises(ConversionDefinitionError, match="duplicate stable id"):
        PresentationAdapterBundle.for_module(
            left_module, "fixture.left", 1
        ).register_into(graph)


def test_explicit_ambiguity_remains_visible_and_never_shape_infers() -> None:
    module = _module(1)
    bundle = PresentationAdapterBundle.for_module(module, "fixture.ambiguous", 1)
    alternate = Conversion.from_callable(
        "fixture.ambiguous.alternate",
        bundle.symbolic_kind,
        bundle.coordinate_kind,
        bundle.coordinates_from_symbolic,
    )
    graph = bundle.graph.with_conversion(alternate)
    with pytest.raises(ConversionAmbiguityError):
        plan_conversion(bundle.symbolic_kind, bundle.coordinate_kind, graph=graph)
    expression = BasisExpression.from_terms(module, (("e0", 1),))
    selected = convert(
        expression,
        bundle.coordinate_kind,
        graph=graph,
        source_kind=bundle.symbolic_kind,
        route=(RouteStep.from_conversion(alternate, "forward"),),
    )
    assert selected.value == expression.to_sparse()


class StringSubclass(str):
    """A rejected non-built-in string declaration."""


class HostileItemsMapping(Mapping[str, object]):
    """Mapping whose ``items`` call leaks only if adapter sanitation is wrong."""

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, object]:
        raise RuntimeError("secret mapping items payload")


class HostileIter:
    """Iterable whose iterator construction raises a private payload."""

    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("secret iter payload")


class HostileNext:
    """Iterator whose next call raises a private payload."""

    def __iter__(self) -> HostileNext:
        return self

    def __next__(self) -> object:
        raise RuntimeError("secret next payload")
