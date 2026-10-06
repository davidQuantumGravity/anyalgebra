"""Exercise literal module parenthood and explicit sparse-element transport.

This bounded example demonstrates infrastructure behavior only. It does not
establish an isomorphism theorem, a basis-change system, or any research claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from anyalgebra.core.domains import DomainElement, ZZ
from anyalgebra.core.elements import (
    ModuleParentMismatchError,
    SparseElement,
    SparseElementConstructionError,
    SparseElementMap,
    transport_sparse_element,
)
from anyalgebra.core.modules import Basis, FreeModule


def _module() -> FreeModule:
    """Construct one deliberately equal-looking, independent free module."""
    return FreeModule(
        ZZ(),
        Basis(("e0", "e1"), coefficient_domain=ZZ()),
        name="coexisting",
    )


def _integer_payload(coefficient: DomainElement[object] | None) -> int:
    """Return one canonical ZZ payload, treating absent sparse entries as zero."""
    if coefficient is None:
        return 0
    if coefficient.parent is not ZZ() or type(coefficient.value) is not int:
        raise TypeError("expected a canonical ZZ coefficient")
    return coefficient.value


def _integer_coordinates(value: SparseElement) -> list[list[int]]:
    """Return canonical ZZ coordinates in a JSON-safe, address-free form."""
    return [
        [index, _integer_payload(coefficient)]
        for index, coefficient in value.coordinates().items()
    ]


def _transformed_coordinates(
    coordinates: Mapping[int, DomainElement[object]],
) -> dict[int, int]:
    """Swap and scale a complete sparse pair, including absent zero entries."""
    return {
        0: -_integer_payload(coordinates.get(1)),
        1: 3 * _integer_payload(coordinates.get(0)),
    }


def _error_name(action: Callable[[], object]) -> str:
    """Run one deliberately rejected action and return its stable error class."""
    try:
        action()
    except (ModuleParentMismatchError, SparseElementConstructionError) as error:
        return type(error).__name__
    raise AssertionError("the declared parent-boundary action unexpectedly succeeded")


def main() -> int:
    """Demonstrate coexistence, rejection, and explicit directional transport."""
    source = _module()
    target = _module()
    foreign = _module()
    source_value = source.element({0: 2, 1: -1})
    target_value = target.element({0: 2, 1: -1})

    def transform(value: SparseElement) -> SparseElement:
        """Apply one explicit non-coordinate-preserving whole-element map."""
        return target.element(_transformed_coordinates(value.coordinates()))

    element_map = SparseElementMap(source, target, transform)
    image = transport_sparse_element(source_value, element_map)
    zero_image = transport_sparse_element(source.zero(), element_map)

    payload = {
        "arithmetic_rejection": _error_name(lambda: source_value.add(target_value)),
        "negative_transport": {
            "foreign_parent": _error_name(
                lambda: transport_sparse_element(
                    foreign.element({0: 2, 1: -1}), element_map
                )
            ),
            "wrong_direction": _error_name(
                lambda: transport_sparse_element(target_value, element_map)
            ),
        },
        "parents": {
            "same_display_name": source.name == target.name,
            "same_fingerprint": source.fingerprint() == target.fingerprint(),
            "same_instance": source is target,
            "structurally_equal": source == target,
        },
        "reinterpretation_rejection": _error_name(lambda: target.element(source_value)),
        "same_coordinates": {
            "python_equal": source_value == target_value,
            "source": _integer_coordinates(source_value),
            "target": _integer_coordinates(target_value),
        },
        "scope": "infrastructure example only; no research claim verified",
        "transport": {
            "coordinate_preserving": _integer_coordinates(image)
            == _integer_coordinates(source_value),
            "image": _integer_coordinates(image),
            "image_parent_is_target": image.parent is target,
            "zero_image": _integer_coordinates(zero_image),
            "zero_parent_is_target": zero_image.parent is target,
        },
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
