"""Contract tests for literal module parents and explicit element transport."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.domains import ZZ
from anyalgebra.core.elements import (
    ModuleParentMismatchError,
    SparseElementConstructionError,
    SparseElementMap,
    SparseElementMapDefinitionError,
    SparseElementTransportError,
    transport_sparse_element,
)
from anyalgebra.core.modules import Basis, FreeModule


def _module(*, name: str = "V") -> FreeModule:
    """Create a deliberately equal-looking but independently owned module."""
    return FreeModule(ZZ(), Basis(("e0", "e1"), coefficient_domain=ZZ()), name=name)


def test_equal_coordinates_do_not_identify_distinct_module_parents() -> None:
    """Coordinate shape, labels, and names never create an implicit transport."""
    source = _module(name="same-label")
    target = _module(name="same-label")
    value = source.element({0: 2, 1: -1})
    target_value = target.element({0: 2, 1: -1})

    assert source == target
    assert value != target_value
    with pytest.raises(SparseElementConstructionError):
        target.element(value)
    with pytest.raises(ModuleParentMismatchError):
        value.add(target_value)


def test_explicit_map_transports_a_whole_element_to_its_declared_target() -> None:
    """A caller-supplied map, not matching coordinates, changes parent ownership."""
    source = _module(name="source")
    target = _module(name="target")
    value = source.element({0: 2, 1: -1})
    element_map = SparseElementMap(
        source,
        target,
        lambda source_value: target.element(
            {
                0: source_value.coordinates()[1],
                1: source_value.coordinates()[0],
            }
        ),
    )

    image = transport_sparse_element(value, element_map)

    assert image.parent is target
    assert image == target.element({0: -1, 1: 2})
    assert tuple(image.coordinates().items()) != tuple(value.coordinates().items())
    assert image != value
    assert image is not value
    assert image.add(target.element({1: 1})) == target.element({0: -1, 1: 3})

    with pytest.raises(ModuleParentMismatchError) as caught:
        transport_sparse_element(target.element({0: 2, 1: -1}), element_map)
    assert caught.value.left_parent is source
    assert caught.value.right_parent is target


def test_explicit_map_rejects_outside_source_before_calling() -> None:
    """A map never infers a source parent from matching coordinates or labels."""
    source = _module(name="source")
    target = _module(name="target")
    foreign = _module(name="foreign")
    calls: list[str] = []

    def record_forward(value: object) -> object:
        calls.append("called")
        assert hasattr(value, "coordinates")
        return target.element(value.coordinates())

    element_map = SparseElementMap(
        source,
        target,
        record_forward,  # type: ignore[arg-type]
    )

    with pytest.raises(ModuleParentMismatchError) as caught:
        transport_sparse_element(foreign.element({0: 3}), element_map)

    assert caught.value.operation == "transport"
    assert caught.value.left_parent is source
    assert caught.value.right_parent is foreign
    assert calls == []


@pytest.mark.parametrize(
    ("forward", "reason"),
    [
        (lambda value: object(), "did not return SparseElement"),
        (lambda value: _module(name="wrong-target").element({}), "declared target"),
    ],
)
def test_explicit_map_validates_its_forward_result_parent(
    forward: object, reason: str
) -> None:
    """A declared map must return an element of its literal declared target."""
    source = _module(name="source")
    target = _module(name="target")
    element_map = SparseElementMap(source, target, forward)  # type: ignore[arg-type]

    with pytest.raises(SparseElementTransportError, match=reason) as caught:
        transport_sparse_element(source.element({}), element_map)

    assert caught.value.element_map is element_map


def test_explicit_map_wraps_forward_failure_and_rejects_malformed_hook() -> None:
    """Transport failures preserve a typed boundary without accepting duck maps."""
    source = _module(name="source")
    target = _module(name="target")

    def broken(value: object) -> object:
        raise RuntimeError("not a mathematical image")

    element_map = SparseElementMap(source, target, broken)  # type: ignore[arg-type]
    with pytest.raises(SparseElementTransportError) as caught:
        transport_sparse_element(source.element({}), element_map)
    assert isinstance(caught.value.__cause__, RuntimeError)

    with pytest.raises(TypeError, match="SparseElementMap"):
        transport_sparse_element(source.element({}), object())  # type: ignore[arg-type]
    with pytest.raises(SparseElementMapDefinitionError, match="source"):
        SparseElementMap(object(), target, lambda value: value)  # type: ignore[arg-type]
    with pytest.raises(SparseElementMapDefinitionError, match="target"):
        SparseElementMap(source, object(), lambda value: value)  # type: ignore[arg-type]
    with pytest.raises(SparseElementMapDefinitionError, match="callable"):
        SparseElementMap(source, target, object())  # type: ignore[arg-type]


def test_explicit_map_is_frozen_unhashable_and_address_free() -> None:
    """The map records declared endpoints without becoming a mutable registry."""
    source = _module(name="source")
    target = _module(name="target")
    element_map = SparseElementMap(source, target, lambda value: value)

    assert repr(element_map) == "SparseElementMap(source_rank=2, target_rank=2)"
    assert "0x" not in repr(element_map)
    with pytest.raises(FrozenInstanceError):
        element_map.source = target  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(element_map)
