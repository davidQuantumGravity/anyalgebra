"""A three-element partial magma without an implicit bottom element."""

from __future__ import annotations

import json
from typing import TypedDict

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.signatures import OperationSymbol


class PartialMagmaExample(TypedDict):
    """The immutable objects making up this finite partial-magma fixture."""

    carrier: FiniteCarrier
    operation: PartialOperation


def build_example() -> PartialMagmaExample:
    """Build a three-element partial multiplication table from public APIs."""
    element = Sort("partial_magma_element")
    carrier = FiniteCarrier(("a", "b", "c"), sort=element)
    product = OperationSymbol("product", (element, element), element)
    undefined_marker = object()
    operation = PartialOperation.from_table(
        product,
        ((element, carrier),),
        (
            (("a", "a"), "a"),
            (("a", "b"), "c"),
            (("b", "a"), "b"),
            (("b", "c"), undefined_marker),
        ),
        undefined_marker=undefined_marker,
    )
    return {"carrier": carrier, "operation": operation}


def run_example() -> dict[str, str]:
    """Exercise both declared and undefined products without totalizing them."""
    operation = build_example()["operation"]
    defined = operation.apply("a", "b")
    explicit_undefined = operation.apply("b", "c")
    omitted_undefined = operation.apply("c", "c")
    if type(defined) is not Defined:
        raise RuntimeError("partial magma declared product was not defined")
    if type(defined.value) is not str:
        raise RuntimeError("partial magma declared product has an unexpected value")
    if type(explicit_undefined) is not Undefined:
        raise RuntimeError("partial magma explicit undefined cell changed meaning")
    if type(omitted_undefined) is not Undefined:
        raise RuntimeError("partial magma omitted cell changed meaning")
    return {
        "defined": defined.value,
        "explicit_undefined": explicit_undefined.reason,
        "omitted_undefined": omitted_undefined.reason,
        "totality": operation.totality,
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
