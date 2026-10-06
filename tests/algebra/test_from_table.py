"""Contract tests for ordered arbitrary-arity basis-product tables."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
    MultilinearDefinitionError,
    StructureConstants,
)
from anyalgebra.core.domains import Domain, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule


def _module(rank: int, *, domain: Domain[object] | None = None) -> FreeModule:
    coefficient_domain = ZZ() if domain is None else domain
    return FreeModule(
        coefficient_domain,
        Basis(
            tuple(f"e{index}" for index in range(rank)),
            coefficient_domain=coefficient_domain,
        ),
    )


def _table(module: FreeModule, arity: int, cells: object) -> BasisProductTable:
    return BasisProductTable.from_cells(module, arity, cast(Any, cells))


def test_binary_table_uses_lexicographic_basis_order_and_matches_constants() -> None:
    module = _module(2)
    table = _table(
        module,
        2,
        (
            {0: ZZ().element(1)},
            {1: ZZ().element(2)},
            {},
            module.element({0: -1, 1: 3}),
        ),
    )
    structure = FiniteMultilinearStructure.from_tables(module, table, name="table")
    direct = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        (
            ((0, 0, 0), ZZ().element(1)),
            ((0, 1, 1), ZZ().element(2)),
            ((1, 1, 0), ZZ().element(-1)),
            ((1, 1, 1), ZZ().element(3)),
        ),
    )

    assert table.arity == 2
    assert len(table.cells) == 4
    assert structure.operation.constants == direct
    assert tuple(structure.evaluate_basis(1, 1).coordinates().items()) == (
        (0, ZZ().element(-1)),
        (1, ZZ().element(3)),
    )


def test_unary_ternary_and_nullary_table_routes_preserve_arity() -> None:
    module = _module(2)
    unary = FiniteMultilinearStructure.from_tables(
        module,
        _table(module, 1, ({0: ZZ().element(5)}, {1: ZZ().element(6)})),
    )
    ternary = FiniteMultilinearStructure.from_tables(
        module,
        _table(module, 3, ({0: ZZ().element(1)},) * 8),
    )
    nullary = FiniteMultilinearStructure.from_tables(
        module, _table(module, 0, ({1: ZZ().element(7)},))
    )

    assert unary.arity == 1
    assert tuple(unary.evaluate_basis(1).coordinates().items()) == (
        (1, ZZ().element(6)),
    )
    assert ternary.arity == 3
    assert tuple(ternary.evaluate_basis(1, 1, 1).coordinates().items()) == (
        (0, ZZ().element(1)),
    )
    assert nullary.arity == 0
    assert tuple(nullary.evaluate_basis().coordinates().items()) == (
        (1, ZZ().element(7)),
    )


def test_rank_zero_input_and_output_boundaries_require_exact_cell_counts() -> None:
    module = _module(0)
    unary = _table(module, 1, ())
    nullary = _table(module, 0, ({},))

    assert (
        FiniteMultilinearStructure.from_tables(
            module, unary
        ).operation.constants.entries
        == ()
    )
    assert (
        FiniteMultilinearStructure.from_tables(module, nullary).evaluate_basis()
        == module.zero()
    )
    with pytest.raises(MultilinearDefinitionError, match="wrong number"):
        _table(module, 1, ({},))
    with pytest.raises(MultilinearDefinitionError, match="wrong number"):
        _table(module, 0, ())


def test_257_ary_singleton_table_compiles_without_dense_allocation() -> None:
    module = _module(1)
    table = _table(module, 257, ({0: ZZ().element(1)},))
    structure = FiniteMultilinearStructure.from_tables(module, (table,))

    assert structure.arity == 257
    assert len(structure.operation.constants.entries) == 1
    assert tuple(structure.evaluate_basis(*((0,) * 257)).coordinates().items()) == (
        (0, ZZ().element(1)),
    )


def test_table_snapshots_generators_and_normalizes_iterator_failure() -> None:
    module = _module(1)
    cells = [{0: ZZ().element(2)}]
    table = _table(module, 1, (cell for cell in cells))
    cells.clear()

    assert tuple(table.cells[0].coordinates().items()) == ((0, ZZ().element(2)),)

    def broken_cells() -> object:
        yield {0: ZZ().element(1)}
        raise RuntimeError("iterator failed")

    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        _table(module, 1, broken_cells())


def test_rejects_malformed_cell_parent_basis_and_count_boundaries() -> None:
    module = _module(1)
    foreign_module = _module(1)
    with pytest.raises(MultilinearDefinitionError, match="exact FreeModule"):
        BasisProductTable.from_cells(object(), 1, ())  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="non-negative"):
        _table(module, True, ())
    with pytest.raises(MultilinearDefinitionError, match="wrong number"):
        _table(module, 1, ())
    with pytest.raises(MultilinearDefinitionError, match="cell"):
        _table(module, 1, (object(),))
    with pytest.raises(MultilinearDefinitionError, match="literal module parent"):
        _table(module, 1, (foreign_module.element({0: 1}),))
    with pytest.raises(MultilinearDefinitionError, match="exact domain element"):
        _table(module, 1, ({0: 1},))
    with pytest.raises(MultilinearDefinitionError, match="literal coefficient parent"):
        _table(module, 1, ({0: QQ().element(1)},))
    with pytest.raises(MultilinearDefinitionError, match="coordinate"):
        _table(module, 1, ({1: ZZ().element(1)},))
    with pytest.raises(MultilinearDefinitionError, match="cell"):
        _table(module, 1, ((module.zero(),),))


def test_from_tables_accepts_exactly_one_table_and_validates_module_and_name() -> None:
    module = _module(1)
    table = _table(module, 1, ({},))
    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_tables(module, ())
    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_tables(module, (table, table))
    with pytest.raises(MultilinearDefinitionError, match="exact BasisProductTable"):
        FiniteMultilinearStructure.from_tables(module, (object(),))  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="literal module"):
        FiniteMultilinearStructure.from_tables(_module(1), table)
    with pytest.raises(MultilinearDefinitionError, match="exact FreeModule"):
        FiniteMultilinearStructure.from_tables(object(), table)  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="name"):
        FiniteMultilinearStructure.from_tables(module, table, name=" ")


def test_operations_and_cells_snapshot_generators_and_normalize_failures() -> None:
    module = _module(1)
    table = _table(module, 1, ({},))
    tables = [table]
    structure = FiniteMultilinearStructure.from_tables(
        module, (candidate for candidate in tables)
    )
    tables.clear()

    assert structure.operation.constants.entries == ()

    def broken_operations() -> Iterator[BasisProductTable]:
        yield table
        raise RuntimeError("operations iterator failed")

    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        FiniteMultilinearStructure.from_tables(module, broken_operations())
    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        _table(module, 1, object())
    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        FiniteMultilinearStructure.from_tables(module, object())  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        _table(module, 1, (_BrokenItemsMapping(),))


def test_invalid_name_preflights_before_operations_iterator_consumption() -> None:
    module = _module(1)
    table = _table(module, 1, ({},))
    untouched = _TouchedIterator(table)

    with pytest.raises(MultilinearDefinitionError, match="name"):
        FiniteMultilinearStructure.from_tables(module, untouched, name=" ")
    assert not untouched.touched
    with pytest.raises(MultilinearDefinitionError, match="name"):
        FiniteMultilinearStructure.from_structure_constants(
            module,
            StructureConstants.from_sparse((module.basis,), module.basis, ()),
            name=" ",
        )


def test_table_record_is_frozen_slotted_unhashable_and_factory_only() -> None:
    module = _module(1)
    table = _table(module, 1, ({},))

    assert not hasattr(table, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(table)
    with pytest.raises(FrozenInstanceError):
        table.cells = ()  # type: ignore[misc]
    with pytest.raises(MultilinearDefinitionError, match="from_cells"):
        BasisProductTable(module, 1, ())


class _BrokenItemsMapping(Mapping[int, object]):
    """A mapping whose item traversal fails after table-cell snapshotting."""

    def __getitem__(self, key: int) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[int, object]:
        raise RuntimeError("mapping items failed")


class _TouchedIterator:
    """One operations iterator that records whether validation consumed it."""

    def __init__(self, table: BasisProductTable) -> None:
        self.table = table
        self.touched = False

    def __iter__(self) -> Iterator[BasisProductTable]:
        self.touched = True
        yield self.table
