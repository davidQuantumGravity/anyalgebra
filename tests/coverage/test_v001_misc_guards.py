"""Close small tensor, map-record, span, and fixture guard seams."""

from __future__ import annotations

from typing import cast

import pytest

import anyalgebra.algebra.tensor as tensor
import anyalgebra.maps.core as maps
from anyalgebra.algebra.multilinear import DeclaredMultilinearCallable
from anyalgebra.algebra.tensor import TensorElement, tensor_product
from anyalgebra.core.domains import Domain, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.fixtures.composition import CompositionFixtureError, _fano_product
from anyalgebra.linear.matrix import MatrixSpace
from anyalgebra.linear.span import SpanError, _outside_witness
from anyalgebra.maps.core import Map, MapDefinitionError, Morphism, MultilinearMap
from anyalgebra.structures.signatures import Signature
from anyalgebra.structures.structure import Structure, StructureBuilder


def _module(label: str = "e") -> FreeModule:
    return FreeModule(ZZ(), Basis((label,), coefficient_domain=ZZ()))


def _structure(name: str = "s") -> Structure:
    sort = Sort(name)
    return (
        StructureBuilder(Signature((sort,)))
        .with_carrier(sort, FiniteCarrier((0,), sort=sort))
        .freeze()
    )


def test_tensor_parent_value_and_equality_private_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    parent = tensor_product(ZZ(), module, max_factors=2)
    assert tensor._value_matches_parent(object(), module) == (False, False)
    assert tensor._value_matches_parent(object(), parent) == (False, False)
    assert tensor._value_matches_parent(object(), object()) == (False, False)
    element = TensorElement.from_factors(parent, (ZZ().element(1), module.zero()))
    assert tensor._value_matches_parent(element, parent) == (True, True)

    class ExplodingMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            del cls, instance
            raise RuntimeError("private")

    class ExplodingProtocol(metaclass=ExplodingMeta):
        pass

    monkeypatch.setattr(tensor, "Domain", ExplodingProtocol)
    assert not tensor._is_declared_parent(object())
    assert tensor._value_matches_parent(object(), cast(Domain[object], object())) == (
        False,
        False,
    )

    class UnknownEquality:
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("private")

    value = UnknownEquality()
    assert tensor._conservatively_equal(value, value)
    assert not tensor._conservatively_equal(value, object())


def test_map_parent_protocol_and_multilinear_identity_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not maps._is_declared_parent(type)

    class ExplodingMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            del cls, instance
            raise RuntimeError("private")

    class ExplodingProtocol(metaclass=ExplodingMeta):
        pass

    monkeypatch.setattr(maps, "Domain", ExplodingProtocol)
    assert not maps._is_declared_parent(object())

    module = _module()
    declaration = DeclaredMultilinearCallable(module, 1, lambda x: x)
    multilinear_map = MultilinearMap.from_callable(
        (module,), module, declaration.function
    )
    assert multilinear_map == multilinear_map


def test_morphism_component_factory_rejects_every_mismatch() -> None:
    source = _structure("s")
    target = _structure("s")
    with pytest.raises(MapDefinitionError, match="source"):
        Morphism.from_components(cast(Structure, object()), target, ())
    with pytest.raises(MapDefinitionError, match="target"):
        Morphism.from_components(source, cast(Structure, object()), ())
    with pytest.raises(MapDefinitionError, match="signature"):
        Morphism.from_components(source, _structure("other"), ())
    with pytest.raises(MapDefinitionError, match="one component"):
        Morphism.from_components(source, target, ())
    with pytest.raises(MapDefinitionError, match="exact Map"):
        Morphism.from_components(source, target, (object(),))

    foreign = _structure("s")
    wrong_endpoint = Map.from_callable(
        foreign.carriers[0], target.carriers[0], lambda x: x
    )
    with pytest.raises(MapDefinitionError, match="literal source and target"):
        Morphism.from_components(source, target, (wrong_endpoint,))


def test_one_sorted_morphism_factory_rejects_bad_endpoints() -> None:
    source = _structure("s")
    target = _structure("s")
    with pytest.raises(MapDefinitionError, match="source"):
        Morphism.from_callable(cast(Structure, object()), target, lambda x: x)
    with pytest.raises(MapDefinitionError, match="target"):
        Morphism.from_callable(source, cast(Structure, object()), lambda x: x)


def test_span_invariant_errors_are_explicit_for_inconsistent_private_inputs() -> None:
    space = MatrixSpace(1, 1, QQ())
    target = space.element(((1,),))
    generator = space.element(((1,),))
    zero_coefficients = space.element(((0,),))
    with pytest.raises(SpanError, match="generator pairing"):
        _outside_witness(target, (generator,), zero_coefficients)
    with pytest.raises(SpanError, match="lacked"):
        _outside_witness(target, (generator,), space.element(((1,),)))

    two_rows = MatrixSpace(2, 1, QQ())
    delayed_target = two_rows.element(((0,), (1,)))
    empty_coefficients = MatrixSpace(2, 0, QQ()).element(((), ()))
    witness = _outside_witness(delayed_target, (), empty_coefficients)
    assert witness.target_pairing.value.numerator != 0


def test_fano_private_precondition_failure_is_explicit() -> None:
    with pytest.raises(CompositionFixtureError, match="omit"):
        _fano_product(1, 1)
