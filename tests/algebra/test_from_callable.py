"""Contract tests for trusted-declaration multilinear callable sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import FrozenInstanceError
import traceback
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    DeclaredMultilinearCallable,
    FiniteMultilinearStructure,
    MultilinearDefinitionError,
    StructureConstants,
)
from anyalgebra.core.domains import Domain, ZZ
from anyalgebra.core.elements import SparseElement
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


def _basis(module: FreeModule, index: int) -> SparseElement:
    return module.element({index: module.domain.element(1)})


def test_binary_callable_order_and_constants_match() -> None:
    module = _module(2)
    calls: list[tuple[int, int]] = []

    def operation(left: SparseElement, right: SparseElement) -> SparseElement:
        assert left.parent is module
        assert right.parent is module
        left_index = next(iter(left.coordinates()))
        right_index = next(iter(right.coordinates()))
        calls.append((left_index, right_index))
        return _basis(module, left_index)

    declaration = DeclaredMultilinearCallable(module, 2, operation)
    structure = FiniteMultilinearStructure.from_callables(module, declaration)
    direct = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            ((left, right, left), ZZ().element(1))
            for left in range(2)
            for right in range(2)
        ),
    )
    table = BasisProductTable.from_cells(
        module,
        2,
        (_basis(module, 0), _basis(module, 0), _basis(module, 1), _basis(module, 1)),
    )
    from_table = FiniteMultilinearStructure.from_tables(module, table)

    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert structure.operation.constants == direct
    assert structure.operation.constants == from_table.operation.constants
    assert structure.evaluate_basis(1, 0) == _basis(module, 1)


def test_unary_ternary_nullary_rank_zero_and_257_ary_boundaries() -> None:
    module = _module(1)
    calls: list[int] = []

    def unary(value: SparseElement) -> SparseElement:
        return value

    def ternary(*values: SparseElement) -> SparseElement:
        return values[0]

    def nullary() -> SparseElement:
        calls.append(0)
        return _basis(module, 0)

    unary_structure = FiniteMultilinearStructure.from_callables(
        module, DeclaredMultilinearCallable(module, 1, unary)
    )
    ternary_structure = FiniteMultilinearStructure.from_callables(
        module, DeclaredMultilinearCallable(module, 3, ternary)
    )
    nullary_structure = FiniteMultilinearStructure.from_callables(
        module, DeclaredMultilinearCallable(module, 0, nullary)
    )
    arity_257 = FiniteMultilinearStructure.from_callables(
        module,
        DeclaredMultilinearCallable(module, 257, lambda *values: values[0]),
    )
    zero_module = _module(0)
    zero_calls: list[int] = []
    rank_zero = FiniteMultilinearStructure.from_callables(
        zero_module,
        DeclaredMultilinearCallable(
            zero_module,
            1,
            lambda value: zero_calls.append(1),
        ),
    )
    zero_nullary_calls: list[int] = []

    def zero_nullary() -> SparseElement:
        zero_nullary_calls.append(1)
        return zero_module.zero()

    rank_zero_nullary = FiniteMultilinearStructure.from_callables(
        zero_module,
        DeclaredMultilinearCallable(
            zero_module,
            0,
            zero_nullary,
        ),
    )

    assert unary_structure.evaluate_basis(0) == _basis(module, 0)
    assert ternary_structure.evaluate_basis(0, 0, 0) == _basis(module, 0)
    assert nullary_structure.evaluate_basis() == _basis(module, 0)
    assert calls == [0]
    assert arity_257.evaluate_basis(*((0,) * 257)) == _basis(module, 0)
    assert rank_zero.operation.constants.entries == ()
    assert zero_calls == []
    assert zero_nullary_calls == [1]
    assert rank_zero_nullary.evaluate_basis() == zero_module.zero()


def test_callable_collection_snapshots_one_item_and_normalizes_failure() -> None:
    module = _module(1)
    declaration = DeclaredMultilinearCallable(module, 1, lambda value: value)
    declarations = [declaration]
    structure = FiniteMultilinearStructure.from_callables(
        module, (item for item in declarations)
    )
    declarations.clear()

    assert structure.arity == 1

    def broken() -> Iterator[DeclaredMultilinearCallable]:
        yield declaration
        raise RuntimeError("iterator failed")

    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        FiniteMultilinearStructure.from_callables(module, broken())


def test_preflight_rejects_metadata_without_invoking_callable() -> None:
    module = _module(1)
    foreign_module = _module(1)
    calls: list[int] = []

    def function(value: SparseElement) -> SparseElement:
        calls.append(1)
        return value

    declaration = DeclaredMultilinearCallable(module, 1, function)
    untouched = _TouchedIterator(declaration)
    with pytest.raises(MultilinearDefinitionError, match="name"):
        FiniteMultilinearStructure.from_callables(module, untouched, name=" ")
    assert not untouched.touched
    assert calls == []
    with pytest.raises(MultilinearDefinitionError, match="non-negative"):
        DeclaredMultilinearCallable(module, True, function)
    with pytest.raises(MultilinearDefinitionError, match="exact FreeModule"):
        DeclaredMultilinearCallable(cast(FreeModule, object()), 1, function)
    with pytest.raises(MultilinearDefinitionError, match="callable"):
        DeclaredMultilinearCallable(module, 1, cast(Callable[..., object], object()))
    with pytest.raises(MultilinearDefinitionError, match="declared_exact"):
        DeclaredMultilinearCallable(module, 1, function, declared_exact=False)
    with pytest.raises(MultilinearDefinitionError, match="declared_exact"):
        DeclaredMultilinearCallable(module, 1, function, declared_exact=cast(bool, 1))
    foreign = DeclaredMultilinearCallable(foreign_module, 1, function)
    with pytest.raises(MultilinearDefinitionError, match="literal module"):
        FiniteMultilinearStructure.from_callables(module, foreign)
    with pytest.raises(MultilinearDefinitionError, match="exact FreeModule"):
        FiniteMultilinearStructure.from_callables(
            cast(FreeModule, object()), declaration
        )
    assert calls == []


def test_callable_collection_requires_exactly_one_exact_declaration() -> None:
    module = _module(1)
    declaration = DeclaredMultilinearCallable(module, 1, lambda value: value)

    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_callables(module, ())
    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_callables(module, (declaration, declaration))
    with pytest.raises(MultilinearDefinitionError, match="exact Declared"):
        FiniteMultilinearStructure.from_callables(
            module, (cast(DeclaredMultilinearCallable, object()),)
        )
    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_callables(
            module,
            cast(Iterable[DeclaredMultilinearCallable], {"only": declaration}),
        )
    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        FiniteMultilinearStructure.from_callables(
            module, cast(Iterable[DeclaredMultilinearCallable], object())
        )


def test_callable_return_errors_are_typed_sanitized_and_parent_checked() -> None:
    module = _module(1)
    foreign = _module(1)
    with pytest.raises(MultilinearDefinitionError, match="exact SparseElement"):
        FiniteMultilinearStructure.from_callables(
            module, DeclaredMultilinearCallable(module, 1, lambda value: object())
        )
    with pytest.raises(MultilinearDefinitionError, match="literal module parent"):
        FiniteMultilinearStructure.from_callables(
            module,
            DeclaredMultilinearCallable(
                module, 1, lambda value: foreign.element({0: 1})
            ),
        )
    with pytest.raises(MultilinearDefinitionError, match="ValueError") as caught:
        FiniteMultilinearStructure.from_callables(
            module,
            DeclaredMultilinearCallable(
                module,
                1,
                lambda value: (_ for _ in ()).throw(ValueError("secret payload")),
            ),
        )
    assert "secret payload" not in str(caught.value)
    assert caught.value.key == (0,)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret payload" not in "".join(traceback.format_exception(caught.value))
    with pytest.raises(MultilinearDefinitionError, match="TypeError") as arity_error:
        FiniteMultilinearStructure.from_callables(
            module, DeclaredMultilinearCallable(module, 2, lambda value: value)
        )
    assert arity_error.value.__cause__ is None
    assert arity_error.value.__context__ is None


def test_callable_record_is_frozen_unhashable_and_identity_only() -> None:
    module = _module(1)

    def function(value: SparseElement) -> SparseElement:
        return value

    left = DeclaredMultilinearCallable(module, 1, function)
    right = DeclaredMultilinearCallable(module, 1, function)

    assert left != right
    assert not hasattr(left, "__dict__")
    assert "0x" not in repr(left)
    with pytest.raises(TypeError, match="unhashable"):
        hash(left)
    with pytest.raises(FrozenInstanceError):
        left.arity = 2  # type: ignore[misc]


class _TouchedIterator:
    def __init__(self, declaration: DeclaredMultilinearCallable) -> None:
        self.declaration = declaration
        self.touched = False

    def __iter__(self) -> Iterator[DeclaredMultilinearCallable]:
        self.touched = True
        yield self.declaration
