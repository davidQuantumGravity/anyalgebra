"""Contract tests for bounded, metadata-only ordered tensor products."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
import traceback
from typing import cast

import pytest

from anyalgebra.algebra.tensor import (
    TensorDefinitionError,
    TensorElement,
    TensorProductParent,
    pure_tensor,
    tensor_product,
)
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule


def _module(label: str) -> FreeModule:
    return FreeModule(ZZ(), Basis((label,), coefficient_domain=ZZ()))


def test_tensor_product_preserves_immediate_factor_order_and_nesting() -> None:
    module_a = _module("a")
    module_b = _module("b")
    module_c = _module("c")
    flat = tensor_product(module_a, module_b, module_c, max_factors=3)
    right_nested = tensor_product(
        module_a, tensor_product(module_b, module_c, max_factors=2), max_factors=2
    )
    left_nested = tensor_product(
        tensor_product(module_a, module_b, max_factors=2), module_c, max_factors=2
    )
    reversed_parent = tensor_product(module_b, module_a, max_factors=2)

    assert flat.factors[0] is module_a
    assert flat.factors[1] is module_b
    assert flat.factors[2] is module_c
    assert right_nested.factors[0] is module_a
    assert right_nested.factors[1] is not module_b
    assert left_nested.factors[0] is not module_a
    assert left_nested.factors[1] is module_c
    assert flat != right_nested
    assert right_nested != left_nested
    assert tensor_product(module_a, module_b, max_factors=2) != reversed_parent
    assert not hasattr(right_nested, "flat_factors")


def test_pure_tensor_accepts_domain_module_and_nested_tensor_values() -> None:
    module = _module("m")
    inner = tensor_product(ZZ(), module, max_factors=2)
    inner_value = pure_tensor(inner, ZZ().element(2), module.element({0: 3}))
    parent = tensor_product(QQ(), inner, max_factors=2)
    value = pure_tensor(parent, QQ().element((1, 2)), inner_value)

    assert value.parent is parent
    assert value.factors[0] == QQ().element((1, 2))
    assert value.factors[1] is inner_value
    assert not hasattr(value, "flattened_factors")
    assert not hasattr(value, "__mul__")


def test_parent_factor_source_is_bounded_and_iterator_errors_are_typed() -> None:
    module = _module("m")
    source = _CountingParents(module)
    with pytest.raises(TensorDefinitionError, match="maximum"):
        TensorProductParent.from_factors(source, max_factors=2)
    assert source.pulls == 3

    def broken() -> Iterator[object]:
        yield ZZ()
        raise RuntimeError("secret iterator payload")

    with pytest.raises(TensorDefinitionError, match="iterated") as caught:
        TensorProductParent.from_factors(broken(), max_factors=3)
    assert "secret iterator payload" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret iterator payload" not in "".join(
        traceback.format_exception(caught.value)
    )
    with pytest.raises(TensorDefinitionError, match="iterable") as initial_caught:
        TensorProductParent.from_factors(_BrokenIterator(), max_factors=3)
    assert initial_caught.value.__cause__ is None
    assert initial_caught.value.__context__ is None
    assert "secret initial payload" not in "".join(
        traceback.format_exception(initial_caught.value)
    )


def test_parent_rejects_invalid_bounds_counts_and_nonparents() -> None:
    module = _module("m")
    untouched = _CountingParents(module)
    with pytest.raises(TensorDefinitionError, match="positive"):
        TensorProductParent.from_factors(untouched, max_factors=0)
    assert untouched.pulls == 0
    with pytest.raises(TensorDefinitionError, match="built-in int"):
        TensorProductParent.from_factors((ZZ(), module), max_factors=True)
    with pytest.raises(TensorDefinitionError, match="at least two"):
        TensorProductParent.from_factors((ZZ(),), max_factors=2)
    with pytest.raises(TensorDefinitionError, match="maximum"):
        TensorProductParent.from_factors((ZZ(), module, QQ()), max_factors=2)
    with pytest.raises(TensorDefinitionError, match="iterable"):
        TensorProductParent.from_factors(
            cast(Iterator[object], object()), max_factors=2
        )
    for invalid in (None, 1, object, _module):
        with pytest.raises(TensorDefinitionError, match="parent"):
            TensorProductParent.from_factors((ZZ(), invalid), max_factors=2)


def test_tensor_element_requires_one_exact_parented_value_per_immediate_factor() -> (
    None
):
    module = _module("m")
    parent = tensor_product(ZZ(), module, max_factors=2)
    foreign_module = _module("m")
    with pytest.raises(TensorDefinitionError, match="wrong number"):
        TensorElement.from_factors(parent, (ZZ().element(1),))
    with pytest.raises(TensorDefinitionError, match="wrong number"):
        TensorElement.from_factors(
            parent, (ZZ().element(1), module.zero(), module.zero())
        )
    overlong_values = _CountingValues(ZZ().element(1))
    with pytest.raises(TensorDefinitionError, match="wrong number"):
        TensorElement.from_factors(parent, overlong_values)
    assert overlong_values.pulls == 3
    with pytest.raises(TensorDefinitionError, match="literal parent"):
        TensorElement.from_factors(parent, (QQ().element(1), module.zero()))
    with pytest.raises(TensorDefinitionError, match="literal parent"):
        TensorElement.from_factors(parent, (ZZ().element(1), foreign_module.zero()))
    with pytest.raises(TensorDefinitionError, match="parented value"):
        TensorElement.from_factors(parent, (1, module.zero()))
    with pytest.raises(TensorDefinitionError, match="exact TensorProductParent"):
        TensorElement.from_factors(cast(TensorProductParent, object()), ())


def test_tensor_element_equality_is_same_parent_only_and_conservative() -> None:
    module = _module("m")
    parent = tensor_product(ZZ(), module, max_factors=2)
    equal_definition = tensor_product(ZZ(), module, max_factors=2)
    first = pure_tensor(parent, ZZ().element(1), module.element({0: 2}))
    second = pure_tensor(parent, ZZ().element(1), module.element({0: 2}))
    distinct_parent = pure_tensor(
        equal_definition, ZZ().element(1), module.element({0: 2})
    )

    assert first == second
    assert first != distinct_parent
    assert parent != equal_definition
    assert not hasattr(parent, "__dict__")
    assert "0x" not in repr(parent)
    with pytest.raises(TypeError, match="unhashable"):
        hash(parent)
    with pytest.raises(FrozenInstanceError):
        parent.factors = ()  # type: ignore[misc]
    assert not hasattr(first, "__dict__")
    assert "0x" not in repr(first)
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)
    with pytest.raises(FrozenInstanceError):
        first.factors = ()  # type: ignore[misc]

    opaque_domain = _OpaqueEqualityDomain()
    opaque_parent = tensor_product(opaque_domain, ZZ(), max_factors=2)
    opaque_left = pure_tensor(opaque_parent, opaque_domain.element(1), ZZ().element(1))
    opaque_right = pure_tensor(opaque_parent, opaque_domain.element(1), ZZ().element(1))
    assert opaque_left != opaque_right


def test_factory_only_records_and_convenience_functions() -> None:
    module = _module("m")
    parent = tensor_product(ZZ(), module, max_factors=2)

    assert pure_tensor(parent, ZZ().element(0), module.zero()).parent is parent
    with pytest.raises(TensorDefinitionError, match="from_factors"):
        TensorProductParent((ZZ(), module))
    with pytest.raises(TensorDefinitionError, match="from_factors"):
        TensorElement(parent, (ZZ().element(0), module.zero()))


class _CountingParents:
    def __init__(self, module: FreeModule) -> None:
        self.module = module
        self.pulls = 0

    def __iter__(self) -> _CountingParents:
        return self

    def __next__(self) -> FreeModule:
        self.pulls += 1
        return self.module


class _CountingValues:
    def __init__(self, value: object) -> None:
        self.value = value
        self.pulls = 0

    def __iter__(self) -> _CountingValues:
        return self

    def __next__(self) -> object:
        self.pulls += 1
        return self.value


class _BrokenIterator:
    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("secret initial payload")


class _OpaqueEqualityDomain:
    """A structural domain whose element equality is not a built-in bool."""

    def normalize(self, value: object) -> int:
        if type(value) is not int:
            raise TypeError("only exact ints are accepted")
        return value

    def element(self, value: object) -> _OpaqueEqualityElement:
        return _OpaqueEqualityElement(self, self.normalize(value))


class _OpaqueEqualityElement:
    def __init__(self, parent: _OpaqueEqualityDomain, value: int) -> None:
        self.parent = parent
        self.value = value

    def __eq__(self, other: object) -> bool:
        del other
        return cast(bool, 1)
