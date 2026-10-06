"""Cover defensive branches at the neutral core boundaries."""

from __future__ import annotations

from typing import cast

import pytest

import anyalgebra.core.domains as domains
import anyalgebra.core.elements as elements
import anyalgebra.core.modules as modules
import anyalgebra.core.parents as parents
from anyalgebra.core.coercions import CoercionMap, CoercionPlan
from anyalgebra.core.domains import Domain, DomainElement, QQ, ZZ
from anyalgebra.core.rational import Rational
from anyalgebra.core.elements import (
    ModuleArithmeticError,
    SparseElementConstructionError,
)
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import (
    CarrierLabelNotFoundError,
    CarrierMemberNotFoundError,
    FiniteCarrier,
    Sort,
)


def _module(rank: int = 2, *, domain: Domain[object] | None = None) -> FreeModule:
    scalar_domain: Domain[object] = ZZ() if domain is None else domain
    return FreeModule(
        scalar_domain,
        Basis(tuple(f"e{i}" for i in range(rank)), coefficient_domain=scalar_domain),
    )


def _edge(
    name: str,
    source: Domain[object],
    target: Domain[object],
    *,
    lossless: bool = True,
) -> CoercionMap[object, object]:
    return CoercionMap(
        name,
        source,
        target,
        lambda value: target.element(value.value),
        injective=lossless,
        exact=True,
        lossless=lossless,
    )


@pytest.mark.parametrize(
    ("sources", "routes", "message"),
    [
        ([], (), "non-empty"),
        ((ZZ(),), [], "match"),
        ((ZZ(),), (), "match"),
        ((ZZ(),), ([],), "exact tuples"),
        ((ZZ(),), ((object(),),), "CoercionMap"),
    ],
)
def test_coercion_plan_rejects_malformed_container_shapes(
    sources: object, routes: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CoercionPlan(
            cast(tuple[Domain[object], ...], sources),
            cast(Domain[object], ZZ()),
            cast(tuple[tuple[CoercionMap[object, object], ...], ...], routes),
        )


def test_coercion_plan_rejects_lossy_and_disconnected_routes() -> None:
    zz = cast(Domain[object], ZZ())
    qq = cast(Domain[object], QQ())
    lossy = _edge("lossy", zz, qq, lossless=False)
    with pytest.raises(ValueError, match="lossless"):
        CoercionPlan((zz,), qq, ((lossy,),))

    disconnected = _edge("disconnected", qq, zz)
    with pytest.raises(ValueError, match="disconnected"):
        CoercionPlan((zz,), zz, ((disconnected,),))
    with pytest.raises(ValueError, match="does not end"):
        CoercionPlan((zz,), qq, ((),))


def test_lossy_common_parent_scan_can_advance_to_a_later_candidate() -> None:
    from anyalgebra.core import coercions

    zz = cast(Domain[object], ZZ())
    qq = cast(Domain[object], QQ())
    lossy = _edge("lossy-later", zz, qq, lossless=False)

    class Paths:
        def paths(
            self, source: Domain[object], candidate: Domain[object]
        ) -> tuple[tuple[CoercionMap[object, object], ...], ...]:
            del source
            return ((),) if candidate is zz else ((lossy,),)

    with pytest.raises(Exception, match="lossy-later"):
        coercions._raise_no_lossless_common_parent(
            cast(coercions.CoercionGraph, Paths()), (zz,), (zz, qq)
        )


def test_domain_rejects_corrupted_private_elements_from_foreign_parents() -> None:
    integer = object.__new__(domains._IntegerElement)
    object.__setattr__(integer, "parent", QQ())
    object.__setattr__(integer, "value", 1)
    with pytest.raises(Exception, match="another parent"):
        ZZ().element(integer)

    rational = object.__new__(domains._RationalElement)
    object.__setattr__(rational, "parent", ZZ())
    object.__setattr__(rational, "value", Rational(1, 1))
    with pytest.raises(Exception, match="another parent"):
        QQ().element(rational)


def test_domain_protocol_recognition_normalizes_instancecheck_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            del cls, instance
            raise RuntimeError("private")

    class ExplodingProtocol(metaclass=ExplodingMeta):
        pass

    monkeypatch.setattr(modules, "Domain", ExplodingProtocol)
    assert not modules._is_domain_instance(object())
    monkeypatch.setattr(parents, "Domain", ExplodingProtocol)
    assert not parents._is_domain_instance(object())
    assert not parents._is_domain_instance(type)


class _ExplodingEquality:
    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError("private")


def test_carrier_membership_preserves_equality_failure_and_label_guard() -> None:
    carrier = FiniteCarrier((_ExplodingEquality(),), sort=Sort("s"))
    with pytest.raises(CarrierMemberNotFoundError, match="equality comparison"):
        assert object() in carrier
    with pytest.raises(CarrierLabelNotFoundError, match="no labels"):
        carrier.label_for(carrier.items[0])


class _Element:
    def __init__(self, parent: object, value: object) -> None:
        self._parent = parent
        self._value = value

    @property
    def parent(self) -> object:
        return self._parent

    @property
    def value(self) -> object:
        return self._value


class _Domain:
    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> object:
        if self.mode == "raise":
            raise RuntimeError("private")
        if self.mode == "invalid":
            return object()
        if isinstance(value, _Element) and value.parent is self:
            return value
        return _Element(self, value)


def test_sparse_right_multiplication_and_subtraction_only_right_coordinate() -> None:
    module = _module()
    left = module.element({1: 2})
    right = module.element({0: 1})
    assert left.__rmul__(right) is NotImplemented
    assert tuple(left.subtract(right).coordinates()) == (0, 1)
    assert tuple(right.subtract(left).coordinates()) == (0, 1)
    assert module.zero().add(right) == right
    assert module.element({0: 2}).subtract(module.element({0: 1})) == module.element(
        {0: 1}
    )
    assert module.element({0: 1}) != module.element({0: 1, 1: 1})


def test_scalar_and_coordinate_construction_reject_invalid_domain_results() -> None:
    domain = cast(Domain[object], _Domain("invalid"))
    module = _module(1, domain=domain)
    with pytest.raises(ModuleArithmeticError, match="did not return"):
        elements._module_scalar(domain, 2, graph=None)
    with pytest.raises(SparseElementConstructionError, match="did not return"):
        module.element({0: 2})


def _corrupt_integer(value: object) -> DomainElement[object]:
    result = object.__new__(domains._IntegerElement)
    object.__setattr__(result, "parent", ZZ())
    object.__setattr__(result, "value", value)
    return cast(DomainElement[object], result)


def test_internal_integer_arithmetic_rejects_corrupt_payloads_and_operation() -> None:
    bad = _corrupt_integer("bad")
    good = cast(DomainElement[object], ZZ().element(1))
    with pytest.raises(ModuleArithmeticError, match="payload"):
        elements._coefficient_binary(
            bad, good, operation="add", coordinate_index=0, domain=ZZ()
        )
    with pytest.raises(ModuleArithmeticError, match="payload"):
        elements._coefficient_unary(
            bad, operation="negate", coordinate_index=0, domain=ZZ()
        )
    with pytest.raises(ModuleArithmeticError, match="unsupported"):
        elements._coefficient_binary(
            good, good, operation="divide", coordinate_index=0, domain=ZZ()
        )


def test_noncallable_coefficient_capability_is_rejected() -> None:
    domain = cast(Domain[object], _Domain())

    class NonCallable(_Element):
        add = 1

    coefficient = cast(DomainElement[object], NonCallable(domain, 1))
    with pytest.raises(ModuleArithmeticError, match="callable add"):
        elements._invoke_coefficient_capability(
            coefficient, "add", coefficient, domain=domain, coordinate_index=0
        )


def test_zero_and_equality_helpers_preserve_uncertainty() -> None:
    raising_domain = cast(Domain[object], _Domain("raise"))
    invalid_domain = cast(Domain[object], _Domain("invalid"))
    assert elements._domain_zero(raising_domain) is None
    assert elements._domain_zero(invalid_domain) is None

    class UnknownEquality(_Element):
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("private")

    domain = cast(Domain[object], _Domain())
    coefficient = cast(DomainElement[object], UnknownEquality(domain, 1))
    zero = cast(DomainElement[object], _Element(domain, 0))
    assert not elements._is_proven_zero(coefficient, None)
    assert not elements._is_proven_zero(coefficient, zero)
    assert not elements._coefficients_equal(coefficient, zero)
