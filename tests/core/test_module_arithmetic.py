"""Contract tests for exact sparse free-module arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from anyalgebra.core.coercions import CoercionGraph, CoercionMap
from anyalgebra.core.domains import Domain, DomainElement, QQ, ZZ
from anyalgebra.core.elements import (
    ModuleArithmeticError,
    ModuleParentMismatchError,
)
from anyalgebra.core.errors import (
    CoercionAmbiguityError,
    CoercionError,
    DomainConstructionError,
    LossyCoercionError,
)
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational


def _module(domain: Domain[object], rank: int) -> FreeModule:
    return FreeModule(
        domain,
        Basis(tuple(f"e{i}" for i in range(rank)), coefficient_domain=domain),
    )


def _zz_to_qq_graph(*, calls: list[str] | None = None) -> CoercionGraph:
    def forward(value: DomainElement[int]) -> DomainElement[Rational]:
        if calls is not None:
            calls.append("coerce")
        return QQ().element(value.value)

    return CoercionGraph().with_map(
        CoercionMap(
            id="test.zz_to_qq",
            source=ZZ(),
            target=QQ(),
            forward=forward,
            injective=True,
            exact=True,
            lossless=True,
        )
    )


def test_module_arithmetic_qq4_exact_fixture_and_named_methods() -> None:
    module = _module(QQ(), 4)
    x = module.element({0: (-1, 2), 2: (3, 4)})
    y = module.element({0: (1, 2), 2: (1, 4)})

    assert x.add(y) == module.element({2: 1})
    assert x.subtract(y) == module.element({0: -1, 2: (1, 2)})
    assert x.negate() == module.element({0: (1, 2), 2: (-3, 4)})
    assert x.scale(2) == module.element({0: -1, 2: (3, 2)})
    assert x + y == x.add(y)
    assert x - y == x.subtract(y)
    assert -x == x.negate()
    assert x * 2 == x.scale(2)
    assert 2 * x == x.scale(2)


def test_module_arithmetic_merges_support_and_removes_cancellation() -> None:
    module = _module(ZZ(), 5)
    left = module.element({4: 8, 0: 2, 2: -3})
    right = module.element({3: 7, 2: 3, 0: 5})
    left_before = tuple(left.coordinates().items())
    right_before = tuple(right.coordinates().items())

    result = left.add(right)

    assert result.parent is module
    assert tuple(result.coordinates().items()) == (
        (0, ZZ().element(7)),
        (3, ZZ().element(7)),
        (4, ZZ().element(8)),
    )
    assert tuple(left.coordinates().items()) == left_before
    assert tuple(right.coordinates().items()) == right_before


def test_module_arithmetic_rank_zero_and_additive_inverse() -> None:
    zero_module = _module(QQ(), 0)
    zero = zero_module.zero()
    assert zero.add(zero) == zero
    assert zero.subtract(zero) == zero
    assert zero.negate() == zero
    assert zero.scale((99, 7)) == zero

    module = _module(ZZ(), 2)
    value = module.element({0: -2, 1: 5})
    assert value.add(value.negate()) == module.zero()


@pytest.mark.parametrize("scalar", [0, 1, -3])
def test_module_scalar_action_accepts_exact_raw_and_owned_zz_scalars(
    scalar: int,
) -> None:
    module = _module(ZZ(), 2)
    value = module.element({0: 2, 1: -4})

    assert value.scale(scalar) == value.scale(ZZ().element(scalar))
    assert value.scale(scalar).parent is module


def test_module_scalar_action_accepts_unique_explicit_foreign_scalar_route() -> None:
    module = _module(QQ(), 2)
    value = module.element({1: (3, 2)})
    integer = ZZ().element(-2)

    assert value.scale(integer, graph=_zz_to_qq_graph()) == module.element({1: -3})


def test_module_scalar_action_accepts_owned_qq_scalar_without_a_graph() -> None:
    module = _module(QQ(), 1)
    value = module.element({0: (2, 3)})

    assert value.scale(QQ().element((3, 4))) == module.element({0: (1, 2)})


def test_module_scalar_action_rejects_foreign_scalar_without_graph() -> None:
    value = _module(QQ(), 1).element({0: 1})

    with pytest.raises(ModuleArithmeticError) as caught:
        value.scale(ZZ().element(2))

    assert caught.value.operation == "scale"
    assert caught.value.coordinate_index is None
    assert "explicit CoercionGraph" in caught.value.reason


def test_module_scalar_action_plans_before_coefficient_operations() -> None:
    domain = _ArithmeticDomain()
    value = _module(domain, 2).element({0: 2, 1: 3})
    scalar_source = _ScalarDomain()
    scalar = scalar_source.element(4)
    calls: list[str] = []

    def forward(raw: DomainElement[int]) -> DomainElement[int]:
        calls.append("coerce")
        return domain.element(raw.value)

    first = CoercionMap(
        id="test.scalar.a",
        source=scalar_source,
        target=domain,
        forward=forward,
        injective=True,
        exact=True,
        lossless=True,
    )
    second = CoercionMap(
        id="test.scalar.b",
        source=scalar_source,
        target=domain,
        forward=forward,
        injective=True,
        exact=True,
        lossless=True,
    )
    with pytest.raises(CoercionAmbiguityError):
        value.scale(scalar, graph=CoercionGraph().with_map(first).with_map(second))
    assert calls == []
    assert domain.operation_calls == []

    lossy = CoercionGraph().with_map(
        CoercionMap(
            id="test.scalar.lossy",
            source=scalar_source,
            target=domain,
            forward=forward,
            exact=True,
        )
    )
    with pytest.raises(LossyCoercionError):
        value.scale(scalar, graph=lossy)
    with pytest.raises(CoercionError):
        value.scale(scalar, graph=CoercionGraph())
    assert calls == []
    assert domain.operation_calls == []


def test_module_scalar_action_rejects_bool_float_and_invalid_graph() -> None:
    value = _module(ZZ(), 1).element({0: 1})

    for scalar in (True, False, 1.0, -2.5):
        with pytest.raises(DomainConstructionError):
            value.scale(scalar)
    with pytest.raises(TypeError, match="graph"):
        value.scale(2, graph=object())  # type: ignore[arg-type]


def test_module_parent_mismatch_uses_literal_identity() -> None:
    left_module = _module(ZZ(), 1)
    right_module = _module(ZZ(), 1)
    assert left_module == right_module
    left = left_module.element({0: 1})
    right = right_module.element({0: 2})

    for operation in (left.add, left.subtract):
        with pytest.raises(ModuleParentMismatchError) as caught:
            operation(right)
        assert caught.value.left_parent is left_module
        assert caught.value.right_parent is right_module
    with pytest.raises(ModuleParentMismatchError):
        _ = left + right
    with pytest.raises(ModuleParentMismatchError):
        _ = left - right


def test_module_arithmetic_rejects_non_elements_and_element_products() -> None:
    value = _module(ZZ(), 1).element({0: 2})

    for operation in (value.add, value.subtract):
        with pytest.raises(ModuleParentMismatchError):
            operation(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = value * value
    with pytest.raises(TypeError):
        _ = value @ value  # type: ignore[operator]
    assert not hasattr(value, "multiply")
    assert not hasattr(value, "product")


def test_module_arithmetic_reports_unsupported_coefficient_capability() -> None:
    domain = _UnsupportedDomain()
    value = _module(domain, 1).element({0: 2})

    invocations: tuple[tuple[str, Callable[[], object]], ...] = (
        ("add", lambda: value.add(value)),
        ("subtract", lambda: value.subtract(value)),
        ("negate", value.negate),
        ("multiply", lambda: value.scale(2)),
    )
    for operation, invoke in invocations:
        with pytest.raises(ModuleArithmeticError) as caught:
            invoke()
        assert caught.value.operation == operation
        assert caught.value.coordinate_index == 0
        assert "capability" in caught.value.reason


def test_module_arithmetic_uses_only_named_custom_domain_capabilities() -> None:
    domain = _ArithmeticDomain()
    module = _module(domain, 2)
    left = module.element({0: 2, 1: 5})
    right = module.element({0: 3, 1: 1})

    assert left.add(right) == module.element({0: 5, 1: 6})
    assert left.subtract(right) == module.element({0: -1, 1: 4})
    assert left.negate() == module.element({0: -2, 1: -5})
    assert left.scale(domain.element(4)) == module.element({0: 8, 1: 20})
    assert domain.operation_calls == [
        "add",
        "add",
        "subtract",
        "subtract",
        "negate",
        "negate",
        "multiply",
        "multiply",
    ]


@pytest.mark.parametrize("mode", ["raise", "wrong_parent", "not_element"])
def test_module_arithmetic_wraps_broken_capability_with_index(mode: str) -> None:
    domain = _ArithmeticDomain(mode=mode)
    value = _module(domain, 1).element({0: 2})

    with pytest.raises(ModuleArithmeticError) as caught:
        value.add(value)

    assert caught.value.operation == "add"
    assert caught.value.coordinate_index == 0
    if mode == "raise":
        assert isinstance(caught.value.__cause__, RuntimeError)
    else:
        assert caught.value.__cause__ is None


def test_module_arithmetic_preserves_exact_rational_normalization() -> None:
    module = _module(QQ(), 1)
    value = module.element({0: (2, 3)})

    assert value.add(value).coordinates()[0].value == Rational(4, 3)
    assert value.scale((9, -6)).coordinates()[0].value == Rational(-1)


@dataclass(frozen=True, slots=True)
class _ScalarElement:
    parent: _ScalarDomain
    value: int


class _ScalarDomain:
    def normalize(self, value: object) -> int:
        if type(value) is not int:
            raise ValueError
        return value

    def element(self, value: object) -> _ScalarElement:
        return _ScalarElement(self, self.normalize(value))


@dataclass(frozen=True, slots=True)
class _UnsupportedElement:
    parent: _UnsupportedDomain
    value: int


class _UnsupportedDomain:
    def normalize(self, value: object) -> int:
        if type(value) is not int:
            raise ValueError
        return value

    def element(self, value: object) -> _UnsupportedElement:
        if type(value) is _UnsupportedElement and value.parent is self:
            return value
        return _UnsupportedElement(self, self.normalize(value))


@dataclass(frozen=True, slots=True)
class _ArithmeticElement:
    parent: _ArithmeticDomain
    value: int

    def add(self, other: object) -> object:
        self.parent.operation_calls.append("add")
        if self.parent.mode == "raise":
            raise RuntimeError("broken arithmetic")
        if self.parent.mode == "wrong_parent":
            return _ArithmeticDomain().element(self.value)
        if self.parent.mode == "not_element":
            return self.value
        assert type(other) is _ArithmeticElement
        return self.parent.element(self.value + other.value)

    def subtract(self, other: object) -> _ArithmeticElement:
        self.parent.operation_calls.append("subtract")
        assert type(other) is _ArithmeticElement
        return self.parent.element(self.value - other.value)

    def negate(self) -> _ArithmeticElement:
        self.parent.operation_calls.append("negate")
        return self.parent.element(-self.value)

    def multiply(self, other: object) -> _ArithmeticElement:
        self.parent.operation_calls.append("multiply")
        assert type(other) is _ArithmeticElement
        return self.parent.element(self.value * other.value)


class _ArithmeticDomain:
    def __init__(self, *, mode: str = "ok") -> None:
        self.mode = mode
        self.operation_calls: list[str] = []

    def normalize(self, value: object) -> int:
        if type(value) is not int:
            raise ValueError
        return value

    def element(self, value: object) -> _ArithmeticElement:
        if type(value) is _ArithmeticElement and value.parent is self:
            return value
        return _ArithmeticElement(self, self.normalize(value))
